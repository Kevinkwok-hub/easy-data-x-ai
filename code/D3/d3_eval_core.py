"""D3 评测的数据加载、校验、确定性指标与分场景聚合。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

RAGAS_METRIC_KEYS = (
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy",
)
RETRIEVAL_METRIC_KEYS = (
    "gold_chunk_recall_at_k",
    "gold_chunk_precision_at_k",
    "top1_reference_hit",
    "evidence_coverage_at_k",
    "all_required_evidence_at_k",
    "stale_evidence_rate",
)
ANCHOR_PATTERNS = (
    r"\bE-\d{4}\b",
    r"\bDBMS_[A-Z_]+\b",
    r"\b[a-z]+(?:_[a-z0-9]+)+\b",
    r"(?<![A-Za-z0-9_])Q[1-4](?![A-Za-z0-9_])",
)
LEXICAL_REWRITES = (
    (r"连接池|连不上|连接相关|连接", "连接"),
    (r"语义搜|关键词搜|混合检索", "混合检索"),
    (r"纯向量|向量搜索|向量检索", "向量检索"),
    (r"付费客户", "付费客户"),
    (r"营收|生意", "营收"),
    (r"升级", "升级"),
    (r"兼容", "兼容"),
    (r"查询性能|太慢|提速|性能", "性能"),
    (r"索引", "索引"),
)
VERSION_PATTERN = r"\bOB-(\d+(?:\.\d+)+)\b"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以稳定格式写入 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_type(case: dict[str, Any]) -> str:
    """兼容 v1 字符串与 v2 对象格式的场景标注。"""
    scenario = case.get("scenario")
    if isinstance(scenario, dict):
        return str(scenario.get("type", ""))
    return str(scenario or "")


def extract_lexical_anchor(question: str) -> str:
    """从用户问题提取短全文查询，不把整句或金标答案送入全文分支。"""
    for pattern in ANCHOR_PATTERNS:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    for pattern, keyword in LEXICAL_REWRITES:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return keyword
    return ""


def extract_metadata_filter(question: str) -> dict[str, str]:
    """单一显式版本号用元数据过滤；多版本问题保留给跨文档检索。"""
    versions = list(dict.fromkeys(re.findall(VERSION_PATTERN, question, re.IGNORECASE)))
    if len(versions) == 1:
        return {"version": versions[0]}
    return {}


def reference_doc_ids(case: dict[str, Any]) -> list[str]:
    """返回 case 所有可接受金标 chunk 的去重列表。"""
    gold = case.get("gold", {})
    ids = list(gold.get("primary_acceptable_chunk_ids", []))
    for fact in gold.get("required_facts", []):
        ids.extend(fact.get("acceptable_chunk_ids", []))
    return list(dict.fromkeys(ids))


def required_facts(case: dict[str, Any]) -> list[dict[str, Any]]:
    """返回需要同时覆盖的事实标注。"""
    return [
        fact
        for fact in case.get("gold", {}).get("required_facts", [])
        if fact.get("required", True)
    ]


def validate_case(
    case: dict[str, Any],
    known_ids: set[str] | None = None,
    top_k: int = 4,
) -> list[str]:
    """返回单个评测样本的结构或金标问题。"""
    errors: list[str] = []
    case_id = case.get("id", "<unknown>")
    for key in ("id", "question", "reference", "scenario", "gold"):
        if key not in case:
            errors.append(f"{case_id}: 缺少字段 {key}")
    facts = required_facts(case)
    if not facts:
        errors.append(f"{case_id}: 缺少 required_facts")
    for fact in facts:
        if not fact.get("acceptable_chunk_ids"):
            errors.append(
                f"{case['id']}: {fact.get('fact_id', '<unknown>')} 没有金标 chunk"
            )
    if scenario_type(case) == "multi_hop" and len(facts) > top_k:
        errors.append(f"{case['id']}: 必需事实数超过 top_k={top_k}")
    if scenario_type(case) not in {"exact", "multi_hop", "temporal", "fuzzy"}:
        errors.append(f"{case['id']}: 未知场景 {scenario_type(case)!r}")
    if "keyword_hint" in case:
        errors.append(f"{case['id']}: keyword_hint 不得进入可执行评测数据")
    if scenario_type(case) == "temporal" and not case.get("temporal", {}).get("as_of"):
        errors.append(f"{case['id']}: 时效题缺少 temporal.as_of")
    if known_ids is not None:
        missing = sorted(set(reference_doc_ids(case)) - known_ids)
        if missing:
            errors.append(f"{case['id']}: 金标 chunk 不存在 {', '.join(missing)}")
    return errors


def validate_dataset(
    dataset: dict[str, Any],
    known_ids: set[str] | None = None,
    top_k: int | None = None,
) -> list[str]:
    """校验 v2 数据集的唯一 ID、场景规模与金标。"""
    errors: list[str] = []
    if dataset.get("schema_version") != "2.0":
        errors.append("dataset schema_version 必须为 2.0")
    cases = dataset.get("cases", [])
    ids = [case.get("id") for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        errors.append("重复 case id: " + ", ".join(str(item) for item in duplicates))
    effective_top_k = int(dataset.get("default_top_k", 4) if top_k is None else top_k)
    if effective_top_k < 1:
        errors.append(f"top_k 必须 >= 1，当前为 {effective_top_k}")
    for case in cases:
        gold = case.get("gold", {})
        if isinstance(gold, dict) and "forbidden_chunk_ids" in gold:
            errors.append(f"{case.get('id', '<unknown>')}: 请使用 temporal.stale_distractor_chunk_ids，勿保留 forbidden_chunk_ids")
        errors.extend(validate_case(case, known_ids=known_ids, top_k=effective_top_k))
    return errors


def compute_retrieval_metrics(
    case: dict[str, Any],
    retrieved_ids: list[str],
) -> dict[str, float]:
    """计算以金标 chunk ID 为准的确定性检索指标。"""
    retrieved = set(retrieved_ids)
    references = set(reference_doc_ids(case))
    matched = references & retrieved
    facts = required_facts(case)
    covered = sum(
        1
        for fact in facts
        if retrieved & set(fact.get("acceptable_chunk_ids", []))
    )
    stale = set(case.get("temporal", {}).get("stale_distractor_chunk_ids", []))
    top1_hit = bool(retrieved_ids and retrieved_ids[0] in references)
    all_required = bool(facts) and covered == len(facts)
    return {
        "gold_chunk_recall_at_k": _ratio(len(matched), len(references)),
        "gold_chunk_precision_at_k": _ratio(len(matched), len(retrieved)),
        "top1_reference_hit": float(top1_hit),
        "evidence_coverage_at_k": _ratio(covered, len(facts)),
        "all_required_evidence_at_k": float(all_required),
        "stale_evidence_rate": _ratio(len(stale & retrieved), len(retrieved)),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def metric_value(row: dict[str, Any], key: str) -> float | None:
    if key in RAGAS_METRIC_KEYS:
        value = row.get("ragas_scores", {}).get(key)
    else:
        value = row.get("retrieval_metrics", {}).get(key)
    if not is_finite_number(value):
        return None
    return float(value)


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    keys: Iterable[str],
) -> dict[str, Any]:
    """聚合运行指标，并记录每个指标的有效样本数。"""
    row_list = list(rows)
    key_list = list(keys)
    metrics: dict[str, float | None] = {}
    n_scored: dict[str, int] = {}
    for key in key_list:
        values = [
            value for value in (metric_value(row, key) for row in row_list) if value is not None
        ]
        n_scored[key] = len(values)
        metrics[key] = round(sum(values) / len(values), 6) if values else None
    return {"n_total": len(row_list), "n_scored": n_scored, "metrics": metrics}


def aggregate_by_scenario(
    rows: Iterable[dict[str, Any]],
    keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("scenario", "unknown"))].append(row)
    return {
        scenario: aggregate_rows(group, keys)
        for scenario, group in sorted(groups.items())
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_metric_comparison(
    vector_rows: Iterable[dict[str, Any]],
    hybrid_rows: Iterable[dict[str, Any]],
    key: str,
    *,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """对同一 case 的指标差值做 bootstrap CI 与配对随机化检验。"""
    if resamples < 100:
        raise ValueError("resamples 必须 >= 100")
    vector_by_id = {str(row.get("case_id")): row for row in vector_rows}
    hybrid_by_id = {str(row.get("case_id")): row for row in hybrid_rows}
    common_ids = sorted(vector_by_id.keys() & hybrid_by_id.keys())
    deltas = []
    for case_id in common_ids:
        vector_value = metric_value(vector_by_id[case_id], key)
        hybrid_value = metric_value(hybrid_by_id[case_id], key)
        if vector_value is not None and hybrid_value is not None:
            deltas.append(hybrid_value - vector_value)
    if not deltas:
        return {
            "n_pairs": 0,
            "mean_delta": None,
            "ci95": None,
            "p_value": None,
            "significant_at_0_05": False,
        }

    observed = sum(deltas) / len(deltas)
    seed = int(sha256_text(key)[:16], 16)
    bootstrap_rng = random.Random(seed)
    bootstrap_means = [
        sum(bootstrap_rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(resamples)
    ]
    ci_low = _percentile(bootstrap_means, 0.025)
    ci_high = _percentile(bootstrap_means, 0.975)

    permutation_rng = random.Random(seed ^ 0x5EED)
    extreme = 0
    for _ in range(resamples):
        permuted = sum(
            delta if permutation_rng.random() < 0.5 else -delta
            for delta in deltas
        ) / len(deltas)
        if abs(permuted) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (resamples + 1)
    significant = p_value < 0.05 and not (ci_low <= 0 <= ci_high)
    return {
        "n_pairs": len(deltas),
        "mean_delta": round(observed, 6),
        "ci95": [round(ci_low, 6), round(ci_high, 6)],
        "p_value": round(p_value, 6),
        "significant_at_0_05": significant,
    }


def compare_strategies(
    vector_rows: Iterable[dict[str, Any]],
    hybrid_rows: Iterable[dict[str, Any]],
    keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """生成所有指标的配对效应量、区间与显著性结果。"""
    vector_list = list(vector_rows)
    hybrid_list = list(hybrid_rows)
    return {
        key: paired_metric_comparison(vector_list, hybrid_list, key)
        for key in keys
    }


def attach_ragas_scores(
    rows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> None:
    """将 RAGAS 返回的逐例 scores 按输入顺序一一对齐。"""
    if len(rows) != len(scores):
        raise ValueError("RAGAS scores 数量与评测行数不一致")
    for row, score in zip(rows, scores):
        row["ragas_scores"] = {
            key: float(value) if is_finite_number(value) else None
            for key, value in score.items()
        }
