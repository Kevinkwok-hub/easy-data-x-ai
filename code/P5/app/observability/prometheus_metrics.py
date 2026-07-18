"""Prometheus metrics for the offline Knowledge Agent API."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram, REGISTRY, generate_latest

from app.schemas import AgentResult


ALLOWED_LABEL_NAMES = {"agent_version", "environment", "status", "tool_name", "error_type"}
DEFAULT_AGENT_VERSION = "mock-v1"
DEFAULT_ENVIRONMENT = "local"


def _env_value(name: str, default: str) -> str:
    """读取低基数环境配置；空字符串按默认值处理，避免 label 出现空维度。"""
    return os.getenv(name, "").strip() or default


def validate_label_names(label_names: tuple[str, ...]) -> None:
    """校验 Prometheus label 是否只使用低基数字段。

    Prometheus label 会参与时间序列维度扩张。教程里明确禁止把 query、answer、user_id、
    trace_id 等高基数或敏感字段放进 label，所以所有指标创建前都通过这个函数兜底。
    """
    forbidden = sorted(set(label_names) - ALLOWED_LABEL_NAMES)
    if forbidden:
        raise ValueError(f"forbidden Prometheus labels: {', '.join(forbidden)}")


class AgentMetrics:
    """集中管理 Agent API 暴露的 Prometheus 指标。

    API 层只告诉这里“请求开始、请求成功、请求异常、Agent 返回了什么结果”。
    具体应该增加哪些 Counter、观察哪个 Histogram，统一收敛在本类中，便于后续
    Grafana Dashboard 和 ROI 映射复用同一套指标名称。
    """

    def __init__(
        self,
        *,
        registry: CollectorRegistry = REGISTRY,
        agent_version: str | None = None,
        environment: str | None = None,
    ) -> None:
        """创建一组指标对象。

        registry 可注入是为了测试隔离：每个测试使用独立 CollectorRegistry，
        不会和 FastAPI 全局 metrics 或其他测试的 Counter 名称冲突。
        """
        self.registry = registry
        self.agent_version = agent_version or _env_value("AGENT_VERSION", DEFAULT_AGENT_VERSION)
        self.environment = environment or _env_value("ENVIRONMENT", DEFAULT_ENVIRONMENT)

        base_labels = ("agent_version", "environment")
        status_labels = ("agent_version", "environment", "status")
        tool_labels = ("agent_version", "environment", "tool_name")
        error_labels = ("agent_version", "environment", "error_type")

        self.requests = self._counter(
            "agent_requests_total",
            "Total Agent API requests grouped by HTTP handling status.",
            status_labels,
        )
        self.errors = self._counter(
            "agent_errors_total",
            "Total Agent API errors grouped by stable error type.",
            error_labels,
        )
        self.duration = self._histogram(
            "agent_request_duration_seconds",
            "Agent API request duration in seconds.",
            status_labels,
        )
        self.token_usage = self._counter(
            "agent_token_usage_total",
            "Total estimated token usage returned by the Agent.",
            base_labels,
        )
        self.cost = self._counter(
            "agent_cost_total",
            "Total mock estimated Agent variable cost in CNY for teaching demos.",
            base_labels,
        )
        self.knowledge_available = self._counter(
            "agent_knowledge_available_total",
            "Total tasks where the local knowledge base had supporting knowledge.",
            base_labels,
        )
        self.knowledge_evaluated = self._counter(
            "agent_knowledge_evaluated_total",
            "Total tasks whose knowledge availability could be evaluated.",
            base_labels,
        )
        self.retrieval = self._counter(
            "agent_retrieval_total",
            "Total tasks evaluated for retrieval metrics.",
            base_labels,
        )
        self.retrieval_hit = self._counter(
            "agent_retrieval_hit_total",
            "Total tasks where retrieval matched a knowledge document.",
            base_labels,
        )
        self.retrieval_errors = self._counter(
            "agent_retrieval_errors_total",
            "Total tasks where the retrieval service failed before a result was available.",
            base_labels,
        )
        self.answer_evaluated = self._counter(
            "agent_answer_evaluated_total",
            "Total tasks with an independent expected-answer annotation.",
            base_labels,
        )
        self.answer_correct = self._counter(
            "agent_answer_correct_total",
            "Total tasks whose answer contains the expected key fact when provided.",
            base_labels,
        )
        self.hallucination = self._counter(
            "agent_hallucination_total",
            "Total tasks marked as hallucinated by the deterministic Agent.",
            base_labels,
        )
        self.tasks = self._counter(
            "agent_tasks_total",
            "Total Agent tasks completed by the API.",
            base_labels,
        )
        self.task_success = self._counter(
            "agent_task_success_total",
            "Total Agent tasks that completed without handoff or hallucination.",
            base_labels,
        )
        self.handoff = self._counter(
            "agent_handoff_total",
            "Total Agent tasks handed off to a human.",
            base_labels,
        )
        self.tool_calls = self._counter(
            "agent_tool_calls_total",
            "Total external tool calls attempted by the Agent.",
            tool_labels,
        )
        self.tool_errors = self._counter(
            "agent_tool_errors_total",
            "Total external tool calls that failed.",
            tool_labels,
        )

    def _counter(self, name: str, description: str, label_names: tuple[str, ...]) -> Counter:
        """创建 Counter 前统一校验 label 白名单。"""
        validate_label_names(label_names)
        return Counter(name, description, label_names, registry=self.registry)

    def _histogram(self, name: str, description: str, label_names: tuple[str, ...]) -> Histogram:
        """创建 Histogram 前统一校验 label 白名单。"""
        validate_label_names(label_names)
        return Histogram(name, description, label_names, registry=self.registry)

    def _base_label_values(self) -> dict[str, str]:
        """返回所有任务级指标共享的低基数 label 值。"""
        return {"agent_version": self.agent_version, "environment": self.environment}

    @contextmanager
    def track_request(self) -> Iterator[None]:
        """记录 API 请求总数和耗时。

        status 表示 HTTP/API 处理结果，而不是 Agent 业务任务是否成功。
        因此工具失败、转人工、知识缺失仍然是 status=success 的请求；真正抛异常才记为 error。
        """
        started_at = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            self.errors.labels(**self._base_label_values(), error_type="unhandled_exception").inc()
            raise
        finally:
            elapsed = time.perf_counter() - started_at
            labels = {**self._base_label_values(), "status": status}
            self.requests.labels(**labels).inc()
            self.duration.labels(**labels).observe(elapsed)

    def record_agent_result(
        self,
        result: AgentResult,
        *,
        expected_answer_contains: str | None = None,
    ) -> None:
        """把一次 AgentResult 映射为 Prometheus Counter。

        这里的 denominator 由 Grafana/PromQL 在查询时组合得到，例如：
        task_success_rate = agent_task_success_total / agent_tasks_total。
        因此本函数只累计原子事件，不直接计算比例。
        """
        base_labels = self._base_label_values()

        self.tasks.labels(**base_labels).inc()
        self.token_usage.labels(**base_labels).inc(result.token_usage)
        self.cost.labels(**base_labels).inc(result.cost)

        if result.retrieval_failed:
            self.retrieval_errors.labels(**base_labels).inc()
        else:
            self.knowledge_evaluated.labels(**base_labels).inc()
            if result.knowledge_available:
                self.knowledge_available.labels(**base_labels).inc()
                # 检索命中率只衡量“知识存在时检索器是否找到”；漏召回进入分母。
                self.retrieval.labels(**base_labels).inc()
                if result.retrieval_hit:
                    self.retrieval_hit.labels(**base_labels).inc()
        if expected_answer_contains is not None:
            self.answer_evaluated.labels(**base_labels).inc()
            if expected_answer_contains in result.answer:
                self.answer_correct.labels(**base_labels).inc()
        if result.hallucinated:
            self.hallucination.labels(**base_labels).inc()
        if result.task_success:
            self.task_success.labels(**base_labels).inc()
        if result.handoff:
            self.handoff.labels(**base_labels).inc()

        if result.tool_called:
            # 当前 Mock Agent 只有一种抽象工具维度；后续接真实工具时可以替换为稳定工具名。
            tool_labels = {**base_labels, "tool_name": "mock_tool"}
            self.tool_calls.labels(**tool_labels).inc()
            if not result.tool_success:
                self.tool_errors.labels(**tool_labels).inc()

    def render_latest(self) -> bytes:
        """生成 Prometheus exposition 格式文本，供 FastAPI `/metrics` 返回。"""
        return generate_latest(self.registry)


metrics = AgentMetrics()
