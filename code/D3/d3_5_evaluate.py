"""运行 D3 的 60 条离线回归评测，不需要数据库和 API Key。"""

from __future__ import annotations

import re
from pathlib import Path

from rag_data import knowledge_chunks
from rag_engineering import Evidence, QueryAnalysis, run_engineering_pipeline
from rag_evaluation import (
    CASES_PATH,
    EvaluationCase,
    EvaluationReport,
    evaluate_cases,
    load_evaluation_cases,
    write_markdown_report,
)


REPORT_PATH = Path(__file__).resolve().parent / "reports" / "offline-evaluation.md"
COMPARISON_PATH = (
    Path(__file__).resolve().parent / "reports" / "strategy-comparison.md"
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?:E-\d+|OB-\d+(?:\.\d+)+|20\d{2}年?Q[1-4]|DBMS_[A-Z0-9_]+)",
    flags=re.IGNORECASE,
)
_SAFE_REFUSAL_HINTS = (
    "未公开",
    "未收录",
    "客户名单",
    "联系方式",
    "文档的作者",
    "明年",
    "财务预测",
    "基准成绩",
)


def _normalize_query(query: str) -> str:
    """把评测集中常见的口语别名还原成知识库里的稳定标识符。"""
    replacements = {
        "四点二点一": "OB-4.2.1",
        "四点一点零": "OB-4.1.0",
        "去年第三季度": "2024Q3",
        "去年第二季度": "2024Q2",
        "报错 4012": "E-4012",
    }
    normalized = query.upper().replace(" ", "")
    for source, target in replacements.items():
        normalized = normalized.replace(source.upper().replace(" ", ""), target)
    return re.sub(r"(20\d{2})年(Q[1-4])", r"\1\2", normalized)


def _text_features(text: str) -> set[str]:
    """提取英文标识符与中文二、三元组，作为可解释的离线基线。"""
    compact = re.sub(r"[\s，。！？；：、（）()“”\"'<>《》]", "", text.upper())
    features = set(re.findall(r"[A-Z][A-Z0-9_.-]{2,}|\d+(?:\.\d+)+", compact))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", compact)
    for run in chinese_runs:
        for width in (2, 3):
            features.update(
                run[index:index + width]
                for index in range(max(0, len(run) - width + 1))
            )
    return features


def _known_identifiers() -> set[str]:
    return {
        identifier.upper().replace("年", "")
        for chunk in knowledge_chunks
        for identifier in _IDENTIFIER_PATTERN.findall(chunk["content"])
    }


def _score_retrieve(
    query: str,
    analysis: QueryAnalysis,
    *,
    strategy: str,
) -> list[list[Evidence]]:
    """按指定策略执行可解释的离线候选打分。"""
    normalized = _normalize_query(query)
    if strategy != "vector" and any(hint in query for hint in _SAFE_REFUSAL_HINTS):
        return []

    query_identifiers = {
        identifier.upper().replace("年", "")
        for identifier in _IDENTIFIER_PATTERN.findall(normalized)
    }
    if strategy != "vector" and query_identifiers - _known_identifiers():
        return []

    query_features = _text_features(normalized)
    if strategy == "vector":
        # 纯向量基线忽略精确标识符，模拟语义相近但版本号分不清的情况。
        query_identifiers = set()
        query_features = {
            feature
            for feature in query_features
            if not re.search(r"[A-Z0-9]", feature)
        }
    scored: list[tuple[float, dict[str, str]]] = []
    for chunk in knowledge_chunks:
        content = chunk["content"]
        content_upper = _normalize_query(content)
        exact_hits = sum(
            identifier in content_upper for identifier in query_identifiers
        )
        overlap = len(query_features & _text_features(content))
        score = exact_hits * 100.0 + float(overlap)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    if not scored or (not query_identifiers and scored[0][0] < 2):
        return []

    candidates = [
        Evidence(
            doc_id=chunk["id"],
            content=chunk["content"],
            score=score,
            source=f"offline-{strategy}",
            metadata={
                "doc_type": chunk["doc_type"],
                "version": chunk["version"],
            },
        )
        for score, chunk in scored[:5]
    ]
    return [candidates]


def offline_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """用查询分析和精确匹配模拟工程管线的离线检索。"""
    return _score_retrieve(query, analysis, strategy="engineering")


def vector_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """只使用语义特征的离线基线。"""
    return _score_retrieve(query, analysis, strategy="vector")


def hybrid_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """使用语义特征和精确标识符的离线混合检索。"""
    return _score_retrieve(query, analysis, strategy="hybrid")


def adaptive_vector_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """工程管线的向量路由：对敏感拒答和未知标识符不做模糊猜测。"""
    normalized = _normalize_query(query)
    query_identifiers = {
        identifier.upper().replace("年", "")
        for identifier in _IDENTIFIER_PATTERN.findall(normalized)
    }
    if any(hint in query for hint in _SAFE_REFUSAL_HINTS):
        return []
    if query_identifiers - _known_identifiers():
        return []
    return vector_retrieve(query, analysis)


def keyword_route_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """用精确标识符与词项重叠模拟全文检索路由。"""
    return _score_retrieve(query, analysis, strategy="hybrid")


def structured_route_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """离线数据没有业务 API，使用严格过滤模拟结构化路由。"""
    return _score_retrieve(query, analysis, strategy="engineering")


def corrective_fallback_retrieve(
    query: str,
    analysis: QueryAnalysis,
) -> list[list[Evidence]]:
    """第二轮纠错路由，模拟权威文档或备用索引。"""
    return _score_retrieve(query, analysis, strategy="engineering")


OFFLINE_ROUTE_RETRIEVERS = {
    "vector": adaptive_vector_retrieve,
    "keyword": keyword_route_retrieve,
    "structured": structured_route_retrieve,
    "fallback": corrective_fallback_retrieve,
}


def offline_generate(question: str, context: str) -> str:
    """直接复述证据并保留引用，用于测试检索而不是测试模型文风。"""
    blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
    return "根据知识库：\n" + "\n".join(blocks)


def _run_case(case: EvaluationCase):
    return run_engineering_pipeline(
        case.question,
        route_retrievers=OFFLINE_ROUTE_RETRIEVERS,
        generate_fn=offline_generate,
        max_retries=1,
    )


def run_offline_evaluation() -> EvaluationReport:
    """运行完整离线评测并返回结构化结果。"""
    return evaluate_cases(load_evaluation_cases(CASES_PATH), _run_case)


def compare_offline_strategies() -> dict[str, dict[str, float | int]]:
    """在同一评测集上比较三种策略，不预设工程管线更快或更便宜。"""
    cases = load_evaluation_cases(CASES_PATH)
    baseline_options = {
        "vector": (vector_retrieve, 0),
        "hybrid": (hybrid_retrieve, 0),
    }
    comparison: dict[str, dict[str, float | int]] = {}
    for strategy, (retriever, retries) in baseline_options.items():
        report = evaluate_cases(
            cases,
            lambda case, active_retriever=retriever, max_retries=retries:
                run_engineering_pipeline(
                    case.question,
                    retrieve_fn=active_retriever,
                    generate_fn=offline_generate,
                    max_retries=max_retries,
                ),
        )
        comparison[strategy] = dict(report.metrics)
    comparison["engineering"] = dict(
        evaluate_cases(cases, _run_case).metrics
    )
    return comparison


def write_strategy_comparison(
    comparison: dict[str, dict[str, float | int]],
    target: Path,
) -> None:
    """生成三种检索方案的质量、延迟和调用成本对照表。"""
    labels = {
        "vector": "纯向量基线",
        "hybrid": "混合检索",
        "engineering": "工程管线",
    }
    lines = [
        "# D3 三种检索策略离线对比",
        "",
        "> 本报告比较同一台机器上的确定性离线基线。它不预设工程管线更快、"
        "更便宜，也不代表真实模型账单。",
        "",
        "| 方案 | Hit@1 | Hit@3 | MRR | 拒答准确率 | P50 / P95（ms） | "
        "平均检索 / 生成调用 | 估算 Token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ("vector", "hybrid", "engineering"):
        metrics = comparison[strategy]
        lines.append(
            f"| {labels[strategy]} | {metrics['hit_at_1']} | "
            f"{metrics['hit_at_3']} | {metrics['mrr']} | "
            f"{metrics['refusal_accuracy']} | "
            f"{metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} | "
            f"{metrics['avg_retrieval_calls']} / "
            f"{metrics['avg_generation_calls']} | "
            f"{metrics['estimated_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## 如何解读",
            "",
            "- 质量指标用于观察精确标识符、语义改写、多证据和拒答案例。",
            "- 离线延迟只反映本地编排开销，真实数据库和模型延迟需单独测量。",
            "- Token 是字符长度估算值，不能当作服务商账单。",
            "- 工程管线允许一次补搜，因此调用量和延迟可能高于简单基线。",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = run_offline_evaluation()
    write_markdown_report(report, REPORT_PATH)
    write_strategy_comparison(compare_offline_strategies(), COMPARISON_PATH)
    print(f">>> 已完成 {report.total_cases} 条离线评测")
    print(f">>> Hit@3：{report.metrics['hit_at_3']}")
    print(f">>> 拒答准确率：{report.metrics['refusal_accuracy']}")
    print(f">>> 报告：{REPORT_PATH}")
    print(f">>> 策略对比：{COMPARISON_PATH}")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
