"""Dataset loading and metric calculation for the offline Knowledge Agent."""

from __future__ import annotations

import json
from pathlib import Path

from app.agent import KnowledgeAgent
from app.schemas import CaseEvaluation, EvalCase, EvaluationReport


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load newline-delimited evaluation cases from disk.

    使用 JSONL 而不是一个大 JSON 数组，方便教程后续追加样本、按行 review 变更，
    也更接近真实评测集和日志样本常见的存储方式。
    """
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {path}") from exc
            cases.append(EvalCase.from_mapping(raw))
    return cases


def evaluate_cases(cases: list[EvalCase], agent: KnowledgeAgent) -> EvaluationReport:
    """Run all cases through the Agent and calculate aggregate metrics.

    评测流程分两步：先保留每条 case 的运行明细，再从明细计算聚合指标。
    这样做比边跑边只累计数字更适合教学，因为读者可以从最终指标回溯到具体失败样本。
    """
    evaluated: list[CaseEvaluation] = []
    for case in cases:
        result = agent.run(case.query, task_id=case.task_id)
        answer_correct = _answer_correct(case, result.answer)
        mismatch_fields = _behavior_mismatches(case, result)
        evaluated.append(
            CaseEvaluation(
                task_id=case.task_id,
                query=case.query,
                category=case.category,
                result=result,
                answer_correct=answer_correct,
                behavior_correct=not mismatch_fields,
                mismatch_fields=mismatch_fields,
            )
        )

    total = len(evaluated)
    metrics = _calculate_metrics(evaluated)
    return EvaluationReport(
        total_cases=total,
        metrics=metrics,
        metric_groups=_group_metrics(metrics),
        cases=tuple(evaluated),
    )


def _answer_correct(case: EvalCase, answer: str) -> bool:
    """用包含关系做离线准确性判断，避免引入另一个 LLM 作为 judge。"""
    # 当前阶段只验证答案是否包含关键事实，不要求逐字一致；这样能容忍模板文字变化，
    # 但仍然能发现回答缺少关键业务信息的问题。
    return case.expected_answer_contains in answer


def _behavior_mismatches(case: EvalCase, result: object) -> tuple[str, ...]:
    """逐字段核对可观测行为，答案事实正确性仍由 answer_correct 独立表达。"""
    expected = {
        "retrieval_failed": case.expected_retrieval_failed,
        "retrieval_hit": case.expected_retrieval_hit,
        "knowledge_available": case.expected_knowledge_available,
        "tool_called": case.expected_tool_called,
        "tool_success": case.expected_tool_success,
        "handoff": case.expected_handoff,
        "task_success": case.expected_task_success,
        "hallucinated": case.expected_hallucinated,
    }
    return tuple(field for field, value in expected.items() if getattr(result, field) != value)


def _rate(numerator: int, denominator: int) -> float:
    """计算比例指标：numerator 是满足条件的样本数，denominator 是该指标适用的样本总数。

    例如检索命中率只看知识存在且检索完成的样本，工具成功率只看真正调用工具的样本。
    把这个函数单独抽出来，是为了让每个指标都显式说明自己的分母口径。
    """
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _average(values: list[int]) -> float:
    """计算运行层均值指标；空列表返回 0，避免空评测集导致报告生成失败。"""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _calculate_metrics(cases: list[CaseEvaluation]) -> dict[str, float]:
    """从单条评测明细聚合原有九项指标及两项诊断增量。

    数据层指标回答“知识有没有覆盖、检索有没有命中”；
    模型层指标回答“答案是否正确、是否出现幻觉”；
    业务层指标回答“任务是否闭环、是否需要人工介入”；
    运行层补充回答“工具是否可靠、延迟和 token 成本大概是多少”。
    """
    total = len(cases)
    evaluable_cases = [case for case in cases if not case.result.retrieval_failed]
    retrieval_cases = [case for case in evaluable_cases if case.result.knowledge_available]
    tool_called_cases = [case for case in cases if case.result.tool_called]

    metrics = {
        # 检索基础设施故障时无法判断覆盖与否，因此必须从覆盖率分母排除。
        "knowledge_coverage_rate": _rate(
            sum(1 for case in evaluable_cases if case.result.knowledge_available), len(evaluable_cases)
        ),
        # 漏召回进入分母；知识缺失和检索服务异常都不进入检索质量分母。
        "retrieval_hit_rate": _rate(
            sum(1 for case in retrieval_cases if case.result.retrieval_hit), len(retrieval_cases)
        ),
        # 回答准确率 numerator：答案包含期望关键事实的样本数；denominator：全部评测样本。
        "answer_accuracy": _rate(sum(1 for case in cases if case.answer_correct), total),
        # 幻觉率衡量“未掌握知识却给确定答案”的风险，越低越适合进入真实客服场景。
        "hallucination_rate": _rate(sum(1 for case in cases if case.result.hallucinated), total),
        # 任务成功率 numerator：无需人工兜底且完成目标的样本数；denominator：全部评测样本。
        "task_success_rate": _rate(sum(1 for case in cases if case.result.task_success), total),
        # 工具成功率只以实际调用工具的任务为 denominator，用来隔离外部动作执行质量。
        "tool_success_rate": _rate(
            sum(1 for case in tool_called_cases if case.result.tool_success), len(tool_called_cases)
        ),
        # 转人工率反映 Agent 无法独立闭环的业务占比，后续可映射到 ROI 的人工处理成本。
        "human_handoff_rate": _rate(sum(1 for case in cases if case.result.handoff), total),
        "behavior_consistency_rate": _rate(sum(1 for case in cases if case.behavior_correct), total),
        "retrieval_failure_rate": _rate(sum(1 for case in cases if case.result.retrieval_failed), total),
        "average_latency_ms": _average([case.result.latency_ms for case in cases]),
        "average_token_usage": _average([case.result.token_usage for case in cases]),
    }
    return metrics


def _group_metrics(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """把扁平指标分成四层，保持报告结构和课程中的治理框架一致。"""
    return {
        "data_layer": {
            "knowledge_coverage_rate": metrics["knowledge_coverage_rate"],
            "retrieval_hit_rate": metrics["retrieval_hit_rate"],
            "retrieval_failure_rate": metrics["retrieval_failure_rate"],
        },
        "model_layer": {
            "answer_accuracy": metrics["answer_accuracy"],
            "hallucination_rate": metrics["hallucination_rate"],
        },
        "business_layer": {
            "task_success_rate": metrics["task_success_rate"],
            "human_handoff_rate": metrics["human_handoff_rate"],
            "behavior_consistency_rate": metrics["behavior_consistency_rate"],
        },
        "runtime_layer": {
            "tool_success_rate": metrics["tool_success_rate"],
            "average_latency_ms": metrics["average_latency_ms"],
            "average_token_usage": metrics["average_token_usage"],
        },
    }
