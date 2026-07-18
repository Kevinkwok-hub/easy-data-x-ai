"""Report writers for Knowledge Agent evaluation output."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas import EvaluationReport


GROUP_TITLES = {
    "data_layer": "数据层",
    "model_layer": "模型层",
    "business_layer": "业务层",
    "runtime_layer": "运行层",
}

METRIC_LABELS = {
    "knowledge_coverage_rate": "知识覆盖率",
    "retrieval_hit_rate": "检索命中率",
    "retrieval_failure_rate": "检索故障率",
    "answer_accuracy": "回答准确率",
    "hallucination_rate": "幻觉率",
    "task_success_rate": "任务成功率",
    "tool_success_rate": "工具成功率",
    "human_handoff_rate": "转人工率",
    "behavior_consistency_rate": "行为一致率",
    "average_latency_ms": "平均延迟（毫秒）",
    "average_token_usage": "平均 Token 使用量",
}


def write_evaluation_reports(report: EvaluationReport, output_dir: Path) -> list[Path]:
    """Write JSON and Markdown reports.

    报告按数据层、模型层、业务层、运行层分组，是为了把问题定位路径和后续治理动作分开：
    数据层看知识覆盖，模型层看回答质量，业务层看闭环效果，运行层看成本与性能。
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # 固定文件名便于 CI、课程讲义和后续 dashboard 脚本直接读取，不需要再做路径发现。
    json_path = directory / "evaluation_report.json"
    markdown_path = directory / "evaluation_report.md"

    # JSON 保留完整机器可读数据；Markdown 面向课堂展示和人工 review。
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return [json_path, markdown_path]


def _markdown(report: EvaluationReport) -> str:
    """把评测报告渲染成适合课程阅读的 Markdown。"""
    lines = [
        "# Knowledge Agent 评测报告",
        "",
        f"评测样本数：{report.total_cases}",
        "",
        "## 指标分组",
        "",
    ]
    for group_name, group_metrics in report.metric_groups.items():
        # 分组渲染让读者先看指标层次，再看具体样本；这比一张长表更容易定位问题来源。
        lines.extend([f"### {GROUP_TITLES[group_name]}", ""])
        lines.extend(["| 指标 | 数值 |", "| --- | ---: |"])
        for metric_name, value in group_metrics.items():
            lines.append(f"| {METRIC_LABELS[metric_name]} | {_format_metric(metric_name, value)} |")
        lines.append("")

    lines.extend(
        [
            "## 样本明细",
            "",
            "| 任务 | 类别 | 命中文档 | 答案正确 | 行为一致 | 差异字段 | 成功 | 转人工 | 幻觉 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in report.cases:
        # 明细表只展示最能解释指标变化的字段，完整答案和成本数据仍保存在 JSON 报告里。
        result = case.result
        lines.append(
            f"| {case.task_id} | {case.category} | {result.matched_doc_id or '-'} | "
            f"{_yes_no(case.answer_correct)} | {_yes_no(case.behavior_correct)} | "
            f"{', '.join(case.mismatch_fields) or '-'} | "
            f"{_yes_no(result.task_success)} | {_yes_no(result.handoff)} | {_yes_no(result.hallucinated)} |"
        )
    return "\n".join(lines) + "\n"


def _format_metric(metric_name: str, value: float) -> str:
    """比例指标显示为百分比，运行层均值保持原单位。"""
    if metric_name.startswith("average_"):
        return f"{value:.2f}"
    return f"{value * 100:.2f}%"


def _yes_no(value: bool) -> str:
    """Markdown 表格中用中文值展示布尔状态，降低非工程读者理解成本。"""
    return "是" if value else "否"
