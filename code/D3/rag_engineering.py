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
class ValidationResult:
    """答案引用校验结果。"""

    is_valid: bool
    citations: tuple[str, ...]
    reason: str


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


def analyze_query(query: str) -> QueryAnalysis:
    """提取精确标识符，并选择纯向量或混合检索策略。"""
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query 不能为空")

    keywords: list[str] = []
    filters: dict[str, str] = {}

    version_match = _VERSION_PATTERN.search(normalized)
    if version_match:
        version = version_match.group(0).upper()
        keywords.append(version)
        filters["version"] = version

    for pattern in (_ERROR_CODE_PATTERN, _QUARTER_PATTERN, _FUNCTION_PATTERN):
        match = pattern.search(normalized)
        if match:
            keyword = match.group(0).upper()
            if keyword not in keywords:
                keywords.append(keyword)

    strategy = "hybrid" if keywords or filters else "vector"
    return QueryAnalysis(
        original_query=query,
        normalized_query=normalized,
        strategy=strategy,
        keywords=tuple(keywords),
        filters=filters,
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


def validate_answer(answer: str, evidence: Sequence[Evidence]) -> ValidationResult:
    """校验回答是否引用了当前上下文中的真实证据。"""
    citations = tuple(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)))
    if not evidence:
        return ValidationResult(False, citations, "没有可用于回答的证据")
    if not citations:
        return ValidationResult(False, (), "回答缺少证据引用")

    known_ids = {item.doc_id for item in evidence}
    unknown = [citation for citation in citations if citation not in known_ids]
    if unknown:
        return ValidationResult(False, citations, f"包含未知引用：{', '.join(unknown)}")
    return ValidationResult(True, citations, "引用均来自当前检索结果")


def rewrite_query(query: str, analysis: QueryAnalysis, retry_count: int) -> str:
    """用确定性规则扩展查询，避免离线测试依赖另一个模型。"""
    exact_terms = " ".join(analysis.keywords)
    suffix = "官方文档 解决方案" if retry_count == 1 else "相关版本 已知问题"
    return " ".join(part for part in (query, exact_terms, suffix) if part).strip()


def _estimate_tokens(text: str) -> int:
    """给课程演示用的粗略 token 估算，不替代模型账单。"""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def run_engineering_pipeline(
    question: str,
    *,
    retrieve_fn: Callable[
        [str, QueryAnalysis],
        Sequence[Sequence[Evidence]] | Sequence[Evidence],
    ],
    generate_fn: Callable[[str, str], str],
    max_retries: int = 1,
    allowed_access_levels: set[str] | None = None,
    context_max_chars: int = 2400,
) -> PipelineResult:
    """执行“分析→检索→融合→生成→校验→重试”的闭环。"""
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0")

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

    while True:
        final_analysis = analyze_query(current_query)
        candidate_groups = retrieve_fn(current_query, final_analysis)
        retrieval_calls += 1
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
            return PipelineResult(
                answer=fallback,
                evidence=[],
                validation=ValidationResult(False, (), "相关证据无访问权限"),
                trace=PipelineTrace(
                    strategy=final_analysis.strategy,
                    retry_count=retry_count,
                    retrieval_calls=retrieval_calls,
                    generation_calls=generation_calls,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    estimated_input_tokens=input_tokens,
                    estimated_output_tokens=output_tokens,
                ),
                status="access_denied",
            )

        if final_evidence:
            context = build_context(final_evidence, max_chars=context_max_chars)
            input_tokens += _estimate_tokens(question + context)
            answer = generate_fn(question, context)
            generation_calls += 1
            output_tokens += _estimate_tokens(answer)
            final_validation = validate_answer(answer, final_evidence)
            if final_validation.is_valid:
                return PipelineResult(
                    answer=answer,
                    evidence=final_evidence,
                    validation=final_validation,
                    trace=PipelineTrace(
                        strategy=final_analysis.strategy,
                        retry_count=retry_count,
                        retrieval_calls=retrieval_calls,
                        generation_calls=generation_calls,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        estimated_input_tokens=input_tokens,
                        estimated_output_tokens=output_tokens,
                    ),
                    status="answered",
                )

        if retry_count >= max_retries:
            break
        retry_count += 1
        current_query = rewrite_query(question, final_analysis, retry_count)

    fallback = "没有找到足够证据，暂时无法回答。你可以补充版本号、错误码或更多背景。"
    return PipelineResult(
        answer=fallback,
        evidence=final_evidence,
        validation=final_validation,
        trace=PipelineTrace(
            strategy=final_analysis.strategy,
            retry_count=retry_count,
            retrieval_calls=retrieval_calls,
            generation_calls=generation_calls,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
        ),
        status="insufficient_evidence",
    )
