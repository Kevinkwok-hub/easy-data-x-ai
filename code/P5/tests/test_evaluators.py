from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.agent import KnowledgeAgent
from app.evaluation.evaluators import evaluate_cases, load_eval_cases
from app.evaluation.report import write_evaluation_reports
from app.schemas import AgentResult, EvalCase


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval_dataset.jsonl"


class KnowledgeAgentTest(unittest.TestCase):
    def test_agent_returns_success_for_known_question(self) -> None:
        result = KnowledgeAgent().run("退款要在多久内提交？")
        self.assertTrue(result.retrieval_hit)
        self.assertTrue(result.knowledge_available)
        self.assertTrue(result.task_success)
        self.assertEqual(result.matched_doc_id, "kb-general-refund")

    def test_agent_handoffs_when_knowledge_missing(self) -> None:
        result = KnowledgeAgent().run("海外仓冷链温控方案怎么配置？")
        self.assertFalse(result.retrieval_hit)
        self.assertTrue(result.handoff)
        self.assertFalse(result.hallucinated)
        self.assertIn("已转人工处理", result.answer)

    def test_retrieval_miss_failure_and_missing_are_distinct(self) -> None:
        agent = KnowledgeAgent()
        miss = agent.run("simulate_retrieval_miss：企业版 SLA 是多少？")
        failure = agent.run("simulate_retrieval_failure：帮我查询 SLA。")
        missing = agent.run("海外仓冷链温控方案怎么配置？")
        self.assertEqual(
            [(x.knowledge_available, x.retrieval_hit, x.retrieval_failed) for x in (miss, failure, missing)],
            [(True, False, False), (False, False, True), (False, False, False)],
        )

    def test_agent_marks_tool_failure(self) -> None:
        result = KnowledgeAgent().run("请导出用量 CSV。")
        self.assertTrue(result.tool_called)
        self.assertFalse(result.tool_success)
        self.assertTrue(result.handoff)
        self.assertFalse(result.task_success)


class EvaluatorTest(unittest.TestCase):
    def test_eval_case_rejects_non_boolean_expected_fields(self) -> None:
        raw = {
            "task_id": "strict-bool",
            "query": "问题",
            "category": "general",
            "expected_answer_contains": "答案",
            "expected_retrieval_hit": "false",
            "expected_knowledge_available": True,
            "expected_tool_called": False,
            "expected_tool_success": False,
            "expected_handoff": False,
            "expected_task_success": True,
            "expected_hallucinated": False,
        }

        with self.assertRaisesRegex((TypeError, ValueError), "bool|布尔"):
            EvalCase.from_mapping(raw)

    def test_evaluator_calculates_retrieval_hit_rate(self) -> None:
        report = evaluate_cases(load_eval_cases(DATASET), KnowledgeAgent())
        self.assertEqual(report.metrics["retrieval_hit_rate"], round(23 / 24, 4))

    def test_evaluator_uses_revised_denominators(self) -> None:
        report = evaluate_cases(load_eval_cases(DATASET), KnowledgeAgent())
        self.assertEqual(report.metrics["knowledge_coverage_rate"], round(24 / 29, 4))
        self.assertEqual(report.metrics["retrieval_failure_rate"], round(1 / 30, 4))
        self.assertEqual(report.metrics["behavior_consistency_rate"], 1.0)

    def test_evaluator_calculates_hallucination_rate(self) -> None:
        report = evaluate_cases(load_eval_cases(DATASET), KnowledgeAgent())
        self.assertEqual(report.metrics["hallucination_rate"], round(1 / 30, 4))

    def test_evaluator_calculates_handoff_rate(self) -> None:
        report = evaluate_cases(load_eval_cases(DATASET), KnowledgeAgent())
        self.assertEqual(report.metrics["human_handoff_rate"], round(7 / 30, 4))

    def test_run_eval_writes_json_and_markdown_reports(self) -> None:
        report = evaluate_cases(load_eval_cases(DATASET), KnowledgeAgent())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_reports(report, Path(directory))
            self.assertEqual({path.name for path in paths}, {"evaluation_report.json", "evaluation_report.md"})

            data = json.loads((Path(directory) / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(data["total_cases"], 30)
            self.assertEqual(
                list(data["metric_groups"]),
                ["data_layer", "model_layer", "business_layer", "runtime_layer"],
            )

            markdown = (Path(directory) / "evaluation_report.md").read_text(encoding="utf-8")
            self.assertIn("### 数据层", markdown)
            self.assertIn("### 模型层", markdown)
            self.assertIn("### 业务层", markdown)
            self.assertIn("### 运行层", markdown)
            self.assertIn("答案正确", markdown)
            self.assertIn("行为一致", markdown)
            self.assertIn("差异字段", markdown)

    def test_each_expected_behavior_field_produces_a_named_mismatch(self) -> None:
        base = load_eval_cases(DATASET)[0]
        correct = KnowledgeAgent().run(base.query)
        fields = (
            "retrieval_failed", "retrieval_hit", "knowledge_available", "tool_called",
            "tool_success", "handoff", "task_success", "hallucinated",
        )

        class FakeAgent:
            def run(self, query: str, task_id: str | None = None) -> AgentResult:
                values = correct.to_dict()
                field = task_id.removeprefix("flip-") if task_id else ""
                values[field] = not values[field]
                return AgentResult(**values)

        for field in fields:
            case = type(base)(**{**base.to_dict(), "task_id": f"flip-{field}"})
            evaluation = evaluate_cases([case], FakeAgent()).cases[0]
            self.assertEqual(evaluation.mismatch_fields, (field,))
            self.assertFalse(evaluation.behavior_correct)


if __name__ == "__main__":
    unittest.main()
