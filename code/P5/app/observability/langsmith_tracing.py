"""Optional LangSmith tracing for the offline Knowledge Agent API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from app.observability.prometheus_metrics import DEFAULT_AGENT_VERSION, DEFAULT_ENVIRONMENT
from app.schemas import AgentResult


DEFAULT_LANGSMITH_PROJECT = "easy-data-x-ai-p5"


class TraceRun(Protocol):
    """LangSmith RunTree 需要用到的最小协议，便于测试中注入 fake run。"""

    metadata: dict[str, Any]

    def end(self, *, outputs: dict[str, Any] | None = None) -> None:
        """结束当前 run；真实 LangSmith SDK 会在这里提交 outputs。"""
        ...


class TraceContext(Protocol):
    """LangSmith trace context manager 的最小协议。"""

    def __enter__(self) -> TraceRun:
        """进入 trace 上下文并返回可写 metadata 的 run 对象。"""
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """退出 trace 上下文；真实 SDK 会处理成功或异常收尾。"""
        ...


TraceFactory = Callable[..., TraceContext]
AgentRunner = Callable[[], AgentResult]


@dataclass(frozen=True)
class LangSmithConfig:
    """LangSmith 开关和低基数元数据配置。

    配置对象独立出来，是为了让默认离线路径足够清楚：只有 tracing=true 且存在 API key
    才会真正创建 LangSmith run；否则所有 trace 调用都是 no-op。
    """

    tracing: bool = False
    api_key: str = ""
    project: str = DEFAULT_LANGSMITH_PROJECT
    agent_version: str = DEFAULT_AGENT_VERSION
    environment: str = DEFAULT_ENVIRONMENT

    @classmethod
    def from_env(cls) -> "LangSmithConfig":
        """从环境变量读取阶段 4 约定的 LangSmith 配置。"""
        return cls(
            tracing=_truthy(os.getenv("LANGSMITH_TRACING", "false")),
            api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
            project=os.getenv("LANGSMITH_PROJECT", "").strip() or DEFAULT_LANGSMITH_PROJECT,
            agent_version=os.getenv("AGENT_VERSION", "").strip() or DEFAULT_AGENT_VERSION,
            environment=os.getenv("ENVIRONMENT", "").strip() or DEFAULT_ENVIRONMENT,
        )

    @property
    def enabled(self) -> bool:
        """只有显式开启 tracing 且有 API key 时才写 LangSmith。"""
        return self.tracing and bool(self.api_key)


class LangSmithTracer:
    """把一次 `/ask` 调用转换为 LangSmith 的教学型 Trace 树。

    当前 Mock Agent 没有真实的检索器、LLM 和工具回调，所以这里根据 AgentResult
    生成结构化子 run。这样课程读者先能看到标准 Trace 形状；后续接真实组件时，
    再把这些子 run 移到真实步骤内部即可。
    """

    def __init__(
        self,
        config: LangSmithConfig | None = None,
        *,
        trace_factory: TraceFactory | None = None,
    ) -> None:
        """创建 tracer。

        trace_factory 是专门为测试和教程可解释性保留的注入点：单元测试可以传入 fake，
        不需要真实 LangSmith API key，也不会产生网络请求。
        """
        self.config = config or LangSmithConfig.from_env()
        self._trace_factory = trace_factory

    @classmethod
    def from_env(cls) -> "LangSmithTracer":
        """使用当前进程环境变量创建 tracer，供 FastAPI 应用启动时调用。"""
        return cls(LangSmithConfig.from_env())

    def trace_agent_run(
        self,
        *,
        query: str,
        task_id: str | None,
        expected_answer_contains: str | None,
        run_agent: AgentRunner,
    ) -> AgentResult:
        """运行 Agent，并在启用时写入 Agent/Retrieval/Generation/Tool/Final Trace。

        query 和 answer 可以进入 LangSmith inputs/outputs，方便排查单条任务；
        但这些内容不会进入 Prometheus label，避免高基数和敏感内容污染监控系统。
        """
        if not self.config.enabled:
            return run_agent()

        trace_factory = self._get_trace_factory()
        base_metadata = self._base_metadata(task_id=task_id, category=None, evaluator_score=None)
        with trace_factory(
            "Agent Trace",
            run_type="chain",
            inputs={"query": query, "task_id": task_id},
            project_name=self.config.project,
            metadata=base_metadata,
        ) as root_run:
            result = run_agent()
            evaluator_score = _evaluator_score(result, expected_answer_contains)
            self._update_metadata(
                root_run,
                category=result.category,
                evaluator_score=evaluator_score,
            )
            self._trace_retrieval(trace_factory, root_run, query, result)
            self._trace_generation(trace_factory, root_run, query, result)
            if result.tool_called:
                self._trace_tool(trace_factory, root_run, result)
            self._trace_final_response(trace_factory, root_run, result, evaluator_score)
            root_run.end(outputs=result.to_dict())
            return result

    def _get_trace_factory(self) -> TraceFactory:
        """获取 LangSmith trace 工厂；默认路径延迟导入 SDK。"""
        if self._trace_factory is not None:
            return self._trace_factory
        # 延迟导入 LangSmith：默认离线运行、无 API key 或未开启 tracing 时，不触碰 SDK 写入路径。
        from langsmith import trace

        return trace

    def _base_metadata(
        self,
        *,
        task_id: str | None,
        category: str | None,
        evaluator_score: float | None,
    ) -> dict[str, Any]:
        """构造 root run 的 metadata。

        root run 刚开始时还不知道最终 category 和 evaluator_score，所以允许它们先为 None；
        Agent 执行结束后再用 `_update_metadata` 回填。
        """
        return {
            "agent_version": self.config.agent_version,
            "environment": self.config.environment,
            "task_id": task_id,
            "category": category,
            "evaluator_score": evaluator_score,
        }

    def _update_metadata(
        self,
        run: TraceRun,
        *,
        category: str,
        evaluator_score: float | None,
    ) -> None:
        """回填 root run metadata。

        LangSmith Trace 的 root 节点是排查入口，必须带最终 category 和 evaluator_score，
        否则课堂演示时只能点进子 run 才能看到任务类型和评测结果。
        """
        run.metadata["agent_version"] = self.config.agent_version
        run.metadata["environment"] = self.config.environment
        run.metadata["category"] = category
        run.metadata["evaluator_score"] = evaluator_score

    def _child_metadata(self, root_run: TraceRun, result: AgentResult, evaluator_score: float | None) -> dict[str, Any]:
        """为子 run 复制稳定 metadata，保证 Trace 树每一层都能单独筛选。"""
        return {
            "agent_version": self.config.agent_version,
            "environment": self.config.environment,
            "task_id": root_run.metadata.get("task_id"),
            "category": result.category,
            "evaluator_score": evaluator_score,
        }

    def _trace_retrieval(
        self,
        trace_factory: TraceFactory,
        root_run: TraceRun,
        query: str,
        result: AgentResult,
    ) -> None:
        """记录检索步骤。

        Retrieval Run 用来回答“Agent 有没有找到知识依据”。即使当前是规则检索，
        也保留 matched_doc_id，方便读者把单条回答回溯到本地知识库。
        """
        metadata = self._child_metadata(root_run, result, root_run.metadata.get("evaluator_score"))
        with trace_factory(
            "Retrieval Run",
            run_type="retriever",
            inputs={"query": query},
            project_name=self.config.project,
            parent=root_run,
            metadata=metadata,
        ) as run:
            run.end(
                outputs={
                    "retrieval_hit": result.retrieval_hit,
                    "knowledge_available": result.knowledge_available,
                    "matched_doc_id": result.matched_doc_id,
                }
            )

    def _trace_generation(
        self,
        trace_factory: TraceFactory,
        root_run: TraceRun,
        query: str,
        result: AgentResult,
    ) -> None:
        """记录生成步骤。

        Generation Run 在真实系统中通常对应 LLM 调用；当前 Mock Agent 没有真实模型，
        所以用最终 answer 和 hallucinated 标记模拟模型层可观察信息。
        """
        metadata = self._child_metadata(root_run, result, root_run.metadata.get("evaluator_score"))
        with trace_factory(
            "Generation Run",
            run_type="llm",
            inputs={"query": query, "matched_doc_id": result.matched_doc_id},
            project_name=self.config.project,
            parent=root_run,
            metadata=metadata,
        ) as run:
            run.end(outputs={"answer": result.answer, "hallucinated": result.hallucinated})

    def _trace_tool(self, trace_factory: TraceFactory, root_run: TraceRun, result: AgentResult) -> None:
        """记录工具调用步骤。

        只有 AgentResult 表示实际调用了工具时才创建 Tool Run；这样普通知识问答不会出现
        空的工具节点，Dashboard 和 Trace 结构也更贴近真实执行路径。
        """
        metadata = self._child_metadata(root_run, result, root_run.metadata.get("evaluator_score"))
        with trace_factory(
            "Tool Run",
            run_type="tool",
            inputs={"tool_name": "mock_tool"},
            project_name=self.config.project,
            parent=root_run,
            metadata=metadata,
        ) as run:
            run.end(outputs={"tool_success": result.tool_success, "handoff": result.handoff})

    def _trace_final_response(
        self,
        trace_factory: TraceFactory,
        root_run: TraceRun,
        result: AgentResult,
        evaluator_score: float | None,
    ) -> None:
        """记录最终响应步骤。

        Final Response 是业务视角的收口节点，集中展示最终 answer、是否成功、
        是否转人工和 evaluator_score，便于从单条 Trace 直接判断用户任务是否闭环。
        """
        metadata = self._child_metadata(root_run, result, evaluator_score)
        with trace_factory(
            "Final Response",
            run_type="chain",
            inputs={"category": result.category},
            project_name=self.config.project,
            parent=root_run,
            metadata=metadata,
        ) as run:
            run.end(
                outputs={
                    "answer": result.answer,
                    "task_success": result.task_success,
                    "handoff": result.handoff,
                    "evaluator_score": evaluator_score,
                }
            )


def _truthy(value: str) -> bool:
    """解析布尔环境变量，兼容常见的 true/yes/on/1 写法。"""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _evaluator_score(result: AgentResult, expected_answer_contains: str | None) -> float | None:
    """把可选 expected_answer_contains 转为 trace metadata 中的简单 evaluator score。"""
    if not expected_answer_contains:
        return None
    return 1.0 if expected_answer_contains in result.answer else 0.0
