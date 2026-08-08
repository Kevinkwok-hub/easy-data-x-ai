"""D3 的可测试 RAG 工程化流水线。

本模块刻意把“检索”和“生成”作为可注入依赖，便于课程读者先用离线
假数据理解流程，再替换成 seekdb 与真实模型。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class QueryAnalysis:
    """查询分析结果。"""

    original_query: str
    normalized_query: str
    strategy: str
    keywords: tuple[str, ...]
    filters: Mapping[str, str]
    intent: str = "lookup"
    rewritten_query: str = ""
    sub_queries: tuple[str, ...] = ()
    routes: tuple[str, ...] = ("vector",)


@dataclass(frozen=True)
class RetrievalPlan:
    """一次自适应检索的可解释执行计划。"""

    queries: tuple[str, ...]
    routes: tuple[str, ...]
    retry_count: int
    reason: str


@dataclass(frozen=True)
class Evidence:
    """统一的候选证据结构。"""

    doc_id: str
    content: str
    score: float
    source: str
    access_level: str = "public"
    metadata: Mapping[str, str] = field(default_factory=dict)
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sources:
            object.__setattr__(self, "sources", (self.source,))


@dataclass(frozen=True)
class AdaptiveRetrievalResult:
    """多查询、多路检索的候选结果与执行信息。"""

    candidate_groups: list[list[Evidence]]
    plan: RetrievalPlan
    calls: int


@dataclass(frozen=True)
class ValidationResult:
    """答案引用校验结果。"""

    is_valid: bool
    citations: tuple[str, ...]
    reason: str
    unsupported_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineTrace:
    """单次流水线的可观测数据。"""

    strategy: str
    retry_count: int
    retrieval_calls: int
    generation_calls: int
    latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    query_history: tuple[str, ...] = ()
    retrieval_routes: tuple[str, ...] = ()
    evidence_grades: tuple[tuple[str, float], ...] = ()
    validation_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """工程化流水线的最终结果。"""

    answer: str
    evidence: list[Evidence]
    validation: ValidationResult
    trace: PipelineTrace
    status: str


_ERROR_CODE_PATTERN = re.compile(r"\bE-\d+\b", flags=re.IGNORECASE)
_VERSION_PATTERN = re.compile(r"\bOB-\d+(?:\.\d+)+\b", flags=re.IGNORECASE)
_QUARTER_PATTERN = re.compile(r"\b(?:20\d{2})?Q[1-4]\b", flags=re.IGNORECASE)
_FUNCTION_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_STRUCTURED_HINTS = (
    "实时",
    "当前",
    "最新",
    "多少",
    "数量",
    "余额",
    "库存",
    "订单",
    "状态",
)
_QUERY_ALIASES = {
    "咋办": "如何处理",
    "怎么弄": "如何操作",
    "连不上": "连接失败",
    "报错": "错误码",
}


def _rewrite_for_retrieval(query: str) -> str:
    """用可审计规则归一少量高频口语表达，保留精确标识符。"""
    rewritten = " ".join(query.strip().split())
    for source, target in _QUERY_ALIASES.items():
        rewritten = rewritten.replace(source, target)
    rewritten = re.sub(
        r"错误码\s*[:：]?\s*(\d{3,6})",
        r"E-\1",
        rewritten,
        flags=re.IGNORECASE,
    )
    return rewritten


def _decompose_query(query: str) -> tuple[str, ...]:
    """把显式并列问题拆成最多三个可独立检索的子问题。"""
    parts = [
        part.strip(" ，,。？?")
        for part in re.split(r"(?:以及|同时|并且)", query)
        if part.strip(" ，,。？?")
    ]
    if len(parts) < 2 or any(len(part) < 4 for part in parts):
        return (query,)
    return tuple(parts[:3])


def analyze_query(query: str) -> QueryAnalysis:
    """改写并拆解问题，提取精确标识符并生成检索路由。"""
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query 不能为空")

    rewritten = _rewrite_for_retrieval(normalized)

    keywords: list[str] = []
    filters: dict[str, str] = {}

    typed_patterns = (
        ("version", _VERSION_PATTERN),
        ("error_code", _ERROR_CODE_PATTERN),
        ("quarter", _QUARTER_PATTERN),
        ("function", _FUNCTION_PATTERN),
    )
    for filter_name, pattern in typed_patterns:
        matches = [match.group(0).upper() for match in pattern.finditer(rewritten)]
        for keyword in matches:
            if keyword not in keywords:
                keywords.append(keyword)
            filters.setdefault(filter_name, keyword)
        if len(matches) > 1:
            filters[f"{filter_name}s"] = ",".join(dict.fromkeys(matches))

    if any(token in rewritten for token in ("对比", "区别", "差异")):
        intent = "comparison"
    elif any(token in rewritten for token in ("如何", "怎么", "步骤", "排查")):
        intent = "procedure"
    elif any(token in rewritten for token in _STRUCTURED_HINTS):
        intent = "structured_lookup"
    else:
        intent = "lookup"

    routes: list[str] = []
    if any(token in rewritten for token in _STRUCTURED_HINTS):
        routes.append("structured")
    if keywords or filters:
        routes.append("keyword")
    routes.append("vector")
    routes = list(dict.fromkeys(routes))

    if len(routes) > 1:
        strategy = "hybrid"
    else:
        strategy = routes[0]
    return QueryAnalysis(
        original_query=query,
        normalized_query=normalized,
        strategy=strategy,
        keywords=tuple(keywords),
        filters=filters,
        intent=intent,
        rewritten_query=rewritten,
        sub_queries=_decompose_query(rewritten),
        routes=tuple(routes),
    )


def build_retrieval_plan(
    analysis: QueryAnalysis,
    *,
    retry_count: int = 0,
    available_routes: Iterable[str] | None = None,
) -> RetrievalPlan:
    """根据查询特征和重试次数生成可执行的自适应检索计划。"""
    if retry_count < 0:
        raise ValueError("retry_count 不能小于 0")

    queries = list(analysis.sub_queries or (analysis.rewritten_query,))
    if retry_count:
        expanded = rewrite_query(
            analysis.original_query,
            analysis,
            retry_count,
        )
        if expanded not in queries:
            queries.append(expanded)

    routes = list(analysis.routes)
    available = set(available_routes or ())
    if retry_count and "fallback" in available:
        routes.append("fallback")
    if available:
        routes = [route for route in routes if route in available]
    routes = list(dict.fromkeys(routes))
    if not routes:
        raise ValueError("没有与查询计划匹配的检索器")

    reason = (
        f"intent={analysis.intent}; strategy={analysis.strategy}; "
        f"exact_terms={len(analysis.keywords)}; retry={retry_count}"
    )
    return RetrievalPlan(
        queries=tuple(dict.fromkeys(queries)),
        routes=tuple(routes),
        retry_count=retry_count,
        reason=reason,
    )


def _flatten_groups(
    candidate_groups: Sequence[Sequence[Evidence]] | Sequence[Evidence],
) -> list[list[Evidence]]:
    if not candidate_groups:
        return []
    first = candidate_groups[0]
    if isinstance(first, Evidence):
        return [list(candidate_groups)]  # type: ignore[arg-type]
    return [list(group) for group in candidate_groups]  # type: ignore[arg-type]


def adaptive_retrieve(
    analysis: QueryAnalysis,
    *,
    retrievers: Mapping[
        str,
        Callable[
            [str, QueryAnalysis],
            Sequence[Sequence[Evidence]] | Sequence[Evidence],
        ],
    ],
    retry_count: int = 0,
) -> AdaptiveRetrievalResult:
    """按检索计划执行多查询、多路召回，保留各路结果供 RRF 融合。"""
    if not retrievers:
        raise ValueError("retrievers 不能为空")

    plan = build_retrieval_plan(
        analysis,
        retry_count=retry_count,
        available_routes=retrievers,
    )
    groups: list[list[Evidence]] = []
    calls = 0
    for planned_query in plan.queries:
        for route in plan.routes:
            route_groups = _flatten_groups(
                retrievers[route](planned_query, analysis)
            )
            calls += 1
            groups.extend(group for group in route_groups if group)
    return AdaptiveRetrievalResult(groups, plan, calls)


def fuse_and_rerank(
    candidate_groups: Sequence[Sequence[Evidence]] | Sequence[Evidence],
    *,
    allowed_access_levels: set[str] | None = None,
    rrf_k: int = 60,
    limit: int = 5,
) -> list[Evidence]:
    """先做权限过滤，再用 RRF 融合多路结果并按文档去重。"""
    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于 0")
    if limit < 1:
        raise ValueError("limit 必须大于 0")

    allowed = {"public"} if allowed_access_levels is None else set(allowed_access_levels)
    combined: dict[str, dict[str, object]] = {}

    for group in _flatten_groups(candidate_groups):
        for rank, item in enumerate(group, start=1):
            if item.access_level not in allowed:
                continue
            entry = combined.setdefault(
                item.doc_id,
                {
                    "item": item,
                    "rrf_score": 0.0,
                    "sources": set(),
                    "best_score": item.score,
                },
            )
            entry["rrf_score"] = float(entry["rrf_score"]) + 1.0 / (rrf_k + rank)
            sources = entry["sources"]
            assert isinstance(sources, set)
            sources.update(item.sources)
            if item.score > float(entry["best_score"]):
                entry["item"] = item
                entry["best_score"] = item.score

    ordered = sorted(
        combined.values(),
        key=lambda entry: (
            -float(entry["rrf_score"]),
            -float(entry["best_score"]),
            -_metadata_number(entry["item"], "source_rank"),
            -_metadata_date(entry["item"], "updated_at"),
            str(getattr(entry["item"], "doc_id")),
        ),
    )
    result: list[Evidence] = []
    for entry in ordered[:limit]:
        item = entry["item"]
        sources = entry["sources"]
        assert isinstance(item, Evidence)
        assert isinstance(sources, set)
        result.append(
            Evidence(
                doc_id=item.doc_id,
                content=item.content,
                score=float(entry["rrf_score"]),
                source=sorted(sources)[0],
                access_level=item.access_level,
                metadata=item.metadata,
                sources=tuple(sorted(sources)),
            )
        )
    return result


def _metadata_number(item: object, key: str) -> float:
    if not isinstance(item, Evidence):
        return 0.0
    try:
        return float(item.metadata.get(key, 0))
    except (TypeError, ValueError):
        return 0.0


def _metadata_date(item: object, key: str) -> int:
    if not isinstance(item, Evidence):
        return 0
    raw_value = str(item.metadata.get(key, ""))
    digits = "".join(character for character in raw_value if character.isdigit())
    return int(digits[:8]) if len(digits) >= 8 else 0


def build_context(evidence: Iterable[Evidence], *, max_chars: int = 2400) -> str:
    """在字符预算内组装带稳定引用 ID 的上下文。"""
    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0")

    blocks: list[str] = []
    current_length = 0
    first_oversized_block = ""
    for item in evidence:
        block = f"[{item.doc_id}] {item.content.strip()}"
        separator_length = 2 if blocks else 0
        if current_length + separator_length + len(block) > max_chars:
            # 跳过过长证据，继续寻找后续能完整放入预算的证据。
            # 如果所有证据都超长，最后再安全截取第一条，避免生成阶段拿到空上下文。
            if not first_oversized_block:
                first_oversized_block = block
            continue
        blocks.append(block)
        current_length += separator_length + len(block)
    if blocks:
        return "\n\n".join(blocks)
    return first_oversized_block[:max_chars]


def grade_evidence(
    question: str,
    evidence: Sequence[Evidence],
    *,
    grader_fn: Callable[[str, Evidence], float],
    threshold: float = 0.5,
) -> tuple[list[Evidence], tuple[tuple[str, float], ...]]:
    """在生成前过滤弱证据，作为 CRAG 的轻量评估节点。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须在 0 到 1 之间")

    accepted: list[Evidence] = []
    grades: list[tuple[str, float]] = []
    for item in evidence:
        score = float(grader_fn(question, item))
        if not 0.0 <= score <= 1.0:
            raise ValueError("证据评分必须在 0 到 1 之间")
        grades.append((item.doc_id, round(score, 4)))
        if score >= threshold:
            accepted.append(item)
    return accepted, tuple(grades)


def _answer_claims(answer: str) -> list[str]:
    """切分答案声明，并把句末引用保留在对应声明内。"""
    claims: list[str] = []
    claim_pattern = re.compile(
        r".*?[。！？!?；;](?:\s*(?:\[[A-Za-z0-9_-]+\]\s*)+)?|.+$"
    )
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {
            "根据知识库：",
            "根据知识库:",
            "答案：",
            "答案:",
            "结论：",
            "结论:",
            "参考资料：",
            "参考资料:",
        }:
            continue
        line_claims = [
            match.group(0).strip()
            for match in claim_pattern.finditer(stripped)
            if match.group(0).strip()
        ]
        # `[doc_id] 段落` 约定引用覆盖该段落；句末引用只覆盖当前声明。
        prefix_match = re.match(
            r"^((?:\[[A-Za-z0-9_-]+\]\s*)+)",
            stripped,
        )
        if prefix_match:
            prefix = prefix_match.group(1).strip()
            line_claims = [
                claim
                if re.search(r"\[[A-Za-z0-9_-]+\]", claim)
                else f"{prefix} {claim}"
                for claim in line_claims
            ]
        claims.extend(line_claims)
    return claims


def validate_answer(
    answer: str,
    evidence: Sequence[Evidence],
    *,
    question: str = "",
    support_fn: Callable[[str, str, Sequence[Evidence]], bool] | None = None,
) -> ValidationResult:
    """校验每条答案声明的引用，并可注入模型做语义支持度判断。"""
    citations = tuple(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)))
    if not evidence:
        return ValidationResult(False, citations, "没有可用于回答的证据")
    if not citations:
        return ValidationResult(False, (), "回答缺少证据引用")

    known_ids = {item.doc_id for item in evidence}
    unknown = [citation for citation in citations if citation not in known_ids]
    if unknown:
        return ValidationResult(False, citations, f"包含未知引用：{', '.join(unknown)}")

    evidence_by_id = {item.doc_id: item for item in evidence}
    unsupported: list[str] = []
    for claim in _answer_claims(answer):
        claim_citations = tuple(
            dict.fromkeys(re.findall(r"\[([A-Za-z0-9_-]+)\]", claim))
        )
        claim_text = re.sub(r"\[[A-Za-z0-9_-]+\]", "", claim).strip(
            " \t-*#，,。.!！?？；;：:"
        )
        if not claim_citations or not claim_text:
            unsupported.append(claim)
            continue
        cited_evidence = [evidence_by_id[citation] for citation in claim_citations]
        if support_fn is not None and not support_fn(
            question,
            claim,
            cited_evidence,
        ):
            unsupported.append(claim)

    if unsupported:
        return ValidationResult(
            False,
            citations,
            "存在缺少引用或证据不支持的声明",
            tuple(unsupported),
        )
    return ValidationResult(True, citations, "每条声明均由当前检索证据支持")


def rewrite_query(query: str, analysis: QueryAnalysis, retry_count: int) -> str:
    """用确定性规则改写和扩展查询，避免离线示例依赖另一个模型。"""
    if retry_count < 0:
        raise ValueError("retry_count 不能小于 0")
    base_query = analysis.rewritten_query or _rewrite_for_retrieval(query)
    if retry_count == 0:
        return base_query

    missing_terms = [
        keyword for keyword in analysis.keywords if keyword not in base_query.upper()
    ]
    suffix = (
        "官方文档 根因 排查 解决方案"
        if retry_count == 1
        else "兼容性 相关版本 已知问题"
    )
    return " ".join((base_query, *missing_terms, suffix)).strip()


def _estimate_tokens(text: str) -> int:
    """给课程演示用的粗略 token 估算，不替代模型账单。"""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def run_engineering_pipeline(
    question: str,
    *,
    generate_fn: Callable[[str, str], str],
    retrieve_fn: Callable[
        [str, QueryAnalysis],
        Sequence[Sequence[Evidence]] | Sequence[Evidence],
    ]
    | None = None,
    route_retrievers: Mapping[
        str,
        Callable[
            [str, QueryAnalysis],
            Sequence[Sequence[Evidence]] | Sequence[Evidence],
        ],
    ]
    | None = None,
    evidence_grader_fn: Callable[[str, Evidence], float] | None = None,
    evidence_grade_threshold: float = 0.5,
    answer_support_fn: Callable[[str, str, Sequence[Evidence]], bool] | None = None,
    max_retries: int = 1,
    allowed_access_levels: set[str] | None = None,
    context_max_chars: int = 2400,
) -> PipelineResult:
    """执行“改写→自适应检索→证据评分→生成→校验→纠错”的闭环。"""
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0")
    if (retrieve_fn is None) == (route_retrievers is None):
        raise ValueError("retrieve_fn 和 route_retrievers 必须且只能提供一个")

    started_at = time.perf_counter()
    current_query = question
    retry_count = 0
    retrieval_calls = 0
    generation_calls = 0
    final_analysis = analyze_query(question)
    final_evidence: list[Evidence] = []
    final_validation = ValidationResult(False, (), "尚未生成回答")
    input_tokens = 0
    output_tokens = 0
    query_history: list[str] = []
    route_history: list[str] = []
    evidence_grades: list[tuple[str, float]] = []
    validation_failures: list[str] = []

    def build_trace() -> PipelineTrace:
        return PipelineTrace(
            strategy=final_analysis.strategy,
            retry_count=retry_count,
            retrieval_calls=retrieval_calls,
            generation_calls=generation_calls,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            query_history=tuple(query_history),
            retrieval_routes=tuple(dict.fromkeys(route_history)),
            evidence_grades=tuple(evidence_grades),
            validation_failures=tuple(validation_failures),
        )

    while True:
        if route_retrievers is not None:
            final_analysis = analyze_query(question)
            adaptive_result = adaptive_retrieve(
                final_analysis,
                retrievers=route_retrievers,
                retry_count=retry_count,
            )
            candidate_groups = adaptive_result.candidate_groups
            retrieval_calls += adaptive_result.calls
            query_history.extend(adaptive_result.plan.queries)
            route_history.extend(adaptive_result.plan.routes)
        else:
            final_analysis = analyze_query(current_query)
            assert retrieve_fn is not None
            candidate_groups = retrieve_fn(current_query, final_analysis)
            retrieval_calls += 1
            query_history.append(current_query)
            route_history.append(final_analysis.strategy)

        raw_groups = _flatten_groups(candidate_groups)
        final_evidence = fuse_and_rerank(
            candidate_groups,
            allowed_access_levels=allowed_access_levels,
        )
        allowed = (
            {"public"}
            if allowed_access_levels is None
            else set(allowed_access_levels)
        )
        raw_items = [item for group in raw_groups for item in group]
        if raw_items and not any(
            item.access_level in allowed for item in raw_items
        ):
            fallback = "检索到了相关资料，但当前身份无权访问。请申请权限或联系资料负责人。"
            final_validation = ValidationResult(False, (), "相关证据无访问权限")
            validation_failures.append(final_validation.reason)
            return PipelineResult(
                answer=fallback,
                evidence=[],
                validation=final_validation,
                trace=build_trace(),
                status="access_denied",
            )

        if final_evidence and evidence_grader_fn is not None:
            final_evidence, grades = grade_evidence(
                question,
                final_evidence,
                grader_fn=evidence_grader_fn,
                threshold=evidence_grade_threshold,
            )
            evidence_grades.extend(grades)
            if not final_evidence:
                final_validation = ValidationResult(
                    False,
                    (),
                    "检索结果未通过相关性评分",
                )
                validation_failures.append(final_validation.reason)

        if not final_evidence and not raw_items:
            final_validation = ValidationResult(False, (), "未检索到候选证据")
            validation_failures.append(final_validation.reason)

        if final_evidence:
            context = build_context(final_evidence, max_chars=context_max_chars)
            input_tokens += _estimate_tokens(question + context)
            answer = generate_fn(question, context)
            generation_calls += 1
            output_tokens += _estimate_tokens(answer)
            final_validation = validate_answer(
                answer,
                final_evidence,
                question=question,
                support_fn=answer_support_fn,
            )
            if final_validation.is_valid:
                return PipelineResult(
                    answer=answer,
                    evidence=final_evidence,
                    validation=final_validation,
                    trace=build_trace(),
                    status="answered",
                )
            validation_failures.append(final_validation.reason)

        if retry_count >= max_retries:
            break
        retry_count += 1
        if route_retrievers is None:
            current_query = rewrite_query(question, final_analysis, retry_count)

    fallback = "没有找到足够证据，暂时无法回答。你可以补充版本号、错误码或更多背景。"
    return PipelineResult(
        answer=fallback,
        evidence=final_evidence,
        validation=final_validation,
        trace=build_trace(),
        status="insufficient_evidence",
    )
