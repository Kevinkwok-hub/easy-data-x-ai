"""Shared dataclasses for the offline Knowledge Agent evaluation flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    """本地知识库条目，保留后续替换为向量检索或数据库时所需的最小字段。

    这里没有直接把知识库设计成自由文本列表，而是显式区分关键词、产品型号、版本、
    错误码和工具调用标记。这样做的教学意义是让读者看到：同样是“检索命中”，
    不同业务字段的匹配优先级和风险是不一样的。
    """

    doc_id: str
    category: str
    answer: str
    keywords: tuple[str, ...]
    product_model: str | None = None
    product_version: str | None = None
    error_code: str | None = None
    requires_tool: bool = False
    tool_success: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "KnowledgeDocument":
        """把 JSON 字典转换成强类型对象，尽早收敛外部数据格式的不确定性。"""
        return cls(
            doc_id=str(raw["doc_id"]),
            category=str(raw["category"]),
            answer=str(raw["answer"]),
            keywords=tuple(str(item) for item in raw.get("keywords", [])),
            product_model=raw.get("product_model"),
            product_version=raw.get("product_version"),
            error_code=raw.get("error_code"),
            requires_tool=bool(raw.get("requires_tool", False)),
            tool_success=bool(raw.get("tool_success", True)),
        )


@dataclass(frozen=True)
class AgentResult:
    """Agent 单次运行结果，字段固定便于评测、报告和后续监控复用。

    字段保持扁平是有意为之：评测、Markdown 报告、JSON 输出和未来 Prometheus 指标
    都可以直接读取这些字段，不需要理解 Agent 内部实现。
    """

    answer: str
    retrieval_hit: bool
    retrieval_failed: bool
    knowledge_available: bool
    tool_called: bool
    tool_success: bool
    handoff: bool
    task_success: bool
    hallucinated: bool
    latency_ms: int
    token_usage: int
    cost: float
    matched_doc_id: str | None
    category: str

    def to_dict(self) -> dict[str, Any]:
        """统一序列化出口，避免报告层直接依赖 dataclass 的内部结构。"""
        return asdict(self)


@dataclass(frozen=True)
class EvalCase:
    """评测样本，显式保存期望行为，避免把测试口径藏在 evaluator 代码里。

    教程里把 expected_* 字段写进数据集，是为了让“什么算成功”可审计。
    evaluator 会逐项校验检索、知识、工具、转人工、任务成功和幻觉期望；
    expected_answer_contains 则独立判断答案关键事实，避免混淆“答对”和“行为符合设计”。
    """

    task_id: str
    query: str
    category: str
    expected_answer_contains: str
    expected_retrieval_hit: bool
    expected_retrieval_failed: bool
    expected_knowledge_available: bool
    expected_tool_called: bool
    expected_tool_success: bool
    expected_handoff: bool
    expected_task_success: bool
    expected_hallucinated: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "EvalCase":
        """从 JSONL 单行构造评测样本；这里集中做字段读取，便于定位坏数据。"""
        def strict_bool(field_name: str, default: bool | None = None) -> bool:
            value = raw.get(field_name, default)
            if type(value) is not bool:
                raise TypeError(f"{field_name} 必须是 bool 布尔值")
            return value

        return cls(
            task_id=str(raw["task_id"]),
            query=str(raw["query"]),
            category=str(raw["category"]),
            expected_answer_contains=str(raw["expected_answer_contains"]),
            expected_retrieval_hit=strict_bool("expected_retrieval_hit"),
            expected_retrieval_failed=strict_bool("expected_retrieval_failed", False),
            expected_knowledge_available=strict_bool("expected_knowledge_available"),
            expected_tool_called=strict_bool("expected_tool_called"),
            expected_tool_success=strict_bool("expected_tool_success"),
            expected_handoff=strict_bool("expected_handoff"),
            expected_task_success=strict_bool("expected_task_success"),
            expected_hallucinated=strict_bool("expected_hallucinated"),
        )

    def to_dict(self) -> dict[str, Any]:
        """保留样本原始期望，方便把评测输入和输出放在同一个报告里排查。"""
        return asdict(self)


@dataclass(frozen=True)
class CaseEvaluation:
    """单条样本的评测明细，用于定位指标异常来自哪条 query。

    聚合指标只能告诉我们“整体变好或变坏”，单条明细才能解释是哪类问题导致变化。
    这也是后续接入 dashboard 时最常用的 drill-down 数据。
    """

    task_id: str
    query: str
    category: str
    result: AgentResult
    answer_correct: bool
    behavior_correct: bool
    mismatch_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """把嵌套的 AgentResult 一并序列化，保证 JSON 报告包含完整运行证据。"""
        data = asdict(self)
        data["result"] = self.result.to_dict()
        return data


@dataclass(frozen=True)
class EvaluationReport:
    """聚合后的评测报告，同时保存明细以便后续接入 dashboard。

    metrics 保存扁平指标，metric_groups 保存教学和报告使用的分层视图。
    两份数据略有重复，但可以减少报告层对指标分组规则的硬编码。
    """

    total_cases: int
    metrics: dict[str, float]
    metric_groups: dict[str, dict[str, float]]
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        """输出稳定 JSON 结构，供测试断言、人工阅读和后续自动化流水线复用。"""
        return {
            "total_cases": self.total_cases,
            "metrics": self.metrics,
            "metric_groups": self.metric_groups,
            "cases": [case.to_dict() for case in self.cases],
        }
