"""D3 离线评测工具：加载案例、计算指标并生成 Markdown 报告。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Sequence

from rag_engineering import PipelineResult


CASES_PATH = Path(__file__).resolve().parent / "data" / "evaluation_cases.jsonl"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    question: str
    expected_doc_ids: tuple[str, ...]
    should_answer: bool


@dataclass(frozen=True)
class EvaluationReport:
    total_cases: int
    metrics: dict[str, float | int]
    failures: tuple[dict[str, object], ...]
    category_metrics: dict[str, dict[str, float | int]]


def load_evaluation_cases(path: Path = CASES_PATH) -> list[EvaluationCase]:
    """读取并校验 JSONL 评测集。"""
    if not path.is_file():
        raise FileNotFoundError(f"评测集不存在：{path}")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象")

        case_id = str(payload.get("case_id", "")).strip()
        category = str(payload.get("category", "")).strip()
        question = str(payload.get("question", "")).strip()
        should_answer = payload.get("should_answer")
        expected_doc_ids = payload.get("expected_doc_ids", [])

        if not case_id or not category or not question:
            raise ValueError(f"第 {line_number} 行缺少必填字段")
        if case_id in seen_ids:
            raise ValueError(f"发现重复案例 ID：{case_id}")
        if not isinstance(should_answer, bool):
            raise ValueError(f"案例 {case_id} 的 should_answer 必须是布尔值")
        if not isinstance(expected_doc_ids, list) or not all(
            isinstance(doc_id, str) and doc_id.strip()
            for doc_id in expected_doc_ids
        ):
            raise ValueError(f"案例 {case_id} 的 expected_doc_ids 格式错误")
        if should_answer and not expected_doc_ids:
            raise ValueError(f"可回答案例 {case_id} 必须提供 expected_doc_ids")

        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                question=question,
                expected_doc_ids=tuple(expected_doc_ids),
                should_answer=should_answer,
            )
        )
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _case_metrics(case: EvaluationCase, result: PipelineResult) -> dict[str, float]:
    retrieved_ids = [item.doc_id for item in result.evidence]
    retrieved = set(retrieved_ids)
    expected = set(case.expected_doc_ids)
    relevant_count = sum(doc_id in expected for doc_id in retrieved_ids)
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    answer_accuracy = float(
        case.should_answer
        and result.status == "answered"
        and result.validation.is_valid
        and expected.issubset(retrieved)
        and expected.issubset(set(result.validation.citations))
    )
    refusal_accuracy = float(
        not case.should_answer and result.status == "insufficient_evidence"
    )
    return {
        "hit_at_1": float(bool(retrieved_ids and retrieved_ids[0] in expected)),
        "hit_at_3": float(any(doc_id in expected for doc_id in retrieved_ids[:3])),
        "mrr": reciprocal_rank,
        "context_precision": (
            relevant_count / len(retrieved_ids) if retrieved_ids else 0.0
        ),
        "context_recall": (
            relevant_count / len(expected) if expected else float(not retrieved_ids)
        ),
        "answer_accuracy": answer_accuracy,
        "refusal_accuracy": refusal_accuracy,
    }


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    run_case: Callable[[EvaluationCase], PipelineResult],
) -> EvaluationReport:
    """运行案例并汇总质量、延迟与调用成本指标。"""
    if not cases:
        raise ValueError("cases 不能为空")

    rows: list[tuple[EvaluationCase, PipelineResult, dict[str, float]]] = []
    failures: list[dict[str, object]] = []
    for case in cases:
        result = run_case(case)
        metrics = _case_metrics(case, result)
        rows.append((case, result, metrics))
        passed = (
            bool(metrics["answer_accuracy"])
            if case.should_answer
            else bool(metrics["refusal_accuracy"])
        )
        if not passed:
            failures.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "expected_doc_ids": list(case.expected_doc_ids),
                    "retrieved_doc_ids": [item.doc_id for item in result.evidence],
                    "status": result.status,
                    "reason": result.validation.reason,
                }
            )

    answer_rows = [row for row in rows if row[0].should_answer]
    refusal_rows = [row for row in rows if not row[0].should_answer]
    quality_source = answer_rows or rows
    latencies = [row[1].trace.latency_ms for row in rows]

    metrics: dict[str, float | int] = {
        key: round(mean(row[2][key] for row in quality_source), 4)
        for key in (
            "hit_at_1",
            "hit_at_3",
            "mrr",
            "context_precision",
            "context_recall",
            "answer_accuracy",
        )
    }
    metrics["refusal_accuracy"] = round(
        mean(row[2]["refusal_accuracy"] for row in refusal_rows)
        if refusal_rows
        else 0.0,
        4,
    )
    metrics.update(
        {
            "latency_p50_ms": round(median(latencies), 2),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
            "avg_retrieval_calls": round(
                mean(row[1].trace.retrieval_calls for row in rows),
                2,
            ),
            "avg_generation_calls": round(
                mean(row[1].trace.generation_calls for row in rows),
                2,
            ),
            "estimated_tokens": sum(
                row[1].trace.estimated_input_tokens
                + row[1].trace.estimated_output_tokens
                for row in rows
            ),
        }
    )

    category_metrics: dict[str, dict[str, float | int]] = {}
    for category in sorted({case.category for case in cases}):
        category_rows = [row for row in rows if row[0].category == category]
        category_metrics[category] = {
            "cases": len(category_rows),
            "pass_rate": round(
                mean(
                    row[2]["answer_accuracy"]
                    if row[0].should_answer
                    else row[2]["refusal_accuracy"]
                    for row in category_rows
                ),
                4,
            ),
        }

    return EvaluationReport(
        total_cases=len(cases),
        metrics=metrics,
        failures=tuple(failures),
        category_metrics=category_metrics,
    )


def write_markdown_report(report: EvaluationReport, target: Path) -> None:
    """把评测结果写成便于课程展示和人工复查的 Markdown。"""
    metric_labels = {
        "hit_at_1": "Hit@1",
        "hit_at_3": "Hit@3",
        "mrr": "MRR",
        "context_precision": "上下文精确率",
        "context_recall": "上下文召回率",
        "answer_accuracy": "可回答案例通过率",
        "refusal_accuracy": "拒答准确率",
        "latency_p50_ms": "延迟 P50（ms）",
        "latency_p95_ms": "延迟 P95（ms）",
        "avg_retrieval_calls": "平均检索调用次数",
        "avg_generation_calls": "平均生成调用次数",
        "estimated_tokens": "估算 Token 总量",
    }
    lines = [
        "# D3 Agentic RAG 离线评测报告",
        "",
        f"- 案例数：{report.total_cases}",
        f"- 失败数：{len(report.failures)}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {metric_labels[key]} | {value} |"
        for key, value in report.metrics.items()
    )
    lines.extend(
        [
            "",
            "## 指标口径",
            "",
            "- Hit@1 / Hit@3：前 1 / 3 条结果中是否出现标注证据。",
            "- MRR：首条正确证据排名的倒数。",
            "- 上下文精确率 / 召回率：检索结果中的相关证据比例与标注证据覆盖率。",
            "- 可回答案例通过率：回答成功、引用校验通过，且检索结果与引用覆盖全部标注证据的比例。",
            "- 拒答准确率：证据不足案例被安全拒答的比例。",
            "- Token 为字符长度推算值，只用于本地方案对比，不等同于供应商账单。",
            "",
            "## 分类结果",
            "",
            "| 分类 | 案例数 | 通过率 |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {category} | {values['cases']} | {values['pass_rate']} |"
        for category, values in report.category_metrics.items()
    )
    lines.extend(["", "## 失败案例", ""])
    if report.failures:
        for failure in report.failures:
            lines.append(
                f"- `{failure['case_id']}` {failure['question']}："
                f"{failure['reason']}（检索到 {failure['retrieved_doc_ids']}）"
            )
    else:
        lines.append("- 无")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
