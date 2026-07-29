import json
import tempfile
import unittest
from pathlib import Path

from rag_engineering import Evidence, PipelineResult, PipelineTrace, ValidationResult
from rag_evaluation import (
    CASES_PATH,
    evaluate_cases,
    load_evaluation_cases,
    write_markdown_report,
)
from d3_5_evaluate import run_offline_evaluation
from d3_5_evaluate import (
    compare_offline_strategies,
    write_strategy_comparison,
)
from rag_data import knowledge_chunks


class EvaluationDatasetTests(unittest.TestCase):
    def test_dataset_has_at_least_sixty_valid_and_unique_cases(self):
        cases = load_evaluation_cases(CASES_PATH)

        self.assertGreaterEqual(len(cases), 60)
        self.assertEqual(len(cases), len({case.case_id for case in cases}))
        self.assertGreaterEqual(len({case.category for case in cases}), 6)
        knowledge_ids = {item["id"] for item in knowledge_chunks}
        for case in cases:
            self.assertTrue(case.question.strip())
            if case.should_answer:
                self.assertTrue(case.expected_doc_ids)
                self.assertTrue(set(case.expected_doc_ids).issubset(knowledge_ids))

    def test_loader_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case_id": "same",
                                "category": "exact",
                                "question": "问题一",
                                "expected_doc_ids": ["kb_001"],
                                "should_answer": True,
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "case_id": "same",
                                "category": "exact",
                                "question": "问题二",
                                "expected_doc_ids": ["kb_002"],
                                "should_answer": True,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复"):
                load_evaluation_cases(path)


class EvaluationMetricTests(unittest.TestCase):
    def test_wrong_evidence_cannot_pass_answer_case(self):
        case = load_evaluation_cases(CASES_PATH)[0]
        result = PipelineResult(
            answer="这是一个格式正确、但证据错误的回答。[kb_wrong]",
            evidence=[Evidence("kb_wrong", "无关证据", 1.0, "offline")],
            validation=ValidationResult(True, ("kb_wrong",), "引用格式正确"),
            trace=PipelineTrace("hybrid", 0, 1, 1, 1.0, 10, 10),
            status="answered",
        )

        report = evaluate_cases([case], lambda _: result)

        self.assertEqual(0.0, report.metrics["answer_accuracy"])
        self.assertEqual(1, len(report.failures))

    def test_offline_baseline_runs_every_case_with_stable_quality_floor(self):
        report = run_offline_evaluation()

        self.assertEqual(60, report.total_cases)
        self.assertGreaterEqual(report.metrics["hit_at_3"], 0.8)
        self.assertEqual(1.0, report.metrics["refusal_accuracy"])

    def test_compares_three_strategies_without_hiding_latency_or_calls(self):
        comparison = compare_offline_strategies()

        self.assertEqual({"vector", "hybrid", "engineering"}, set(comparison))
        for metrics in comparison.values():
            self.assertIn("hit_at_3", metrics)
            self.assertIn("latency_p95_ms", metrics)
            self.assertIn("avg_retrieval_calls", metrics)
            self.assertIn("estimated_tokens", metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "comparison.md"
            write_strategy_comparison(comparison, target)
            content = target.read_text(encoding="utf-8")

        self.assertIn("纯向量基线", content)
        self.assertIn("混合检索", content)
        self.assertIn("工程管线", content)
        self.assertIn("不预设", content)

    def test_reports_quality_latency_and_cost_metrics(self):
        cases = load_evaluation_cases(CASES_PATH)[:2]

        def run_case(case):
            evidence = [
                Evidence(doc_id, f"{doc_id} 的证据", 1.0, "offline")
                for doc_id in case.expected_doc_ids
            ]
            answer = (
                "".join(f"[{doc_id}]" for doc_id in case.expected_doc_ids)
                if case.should_answer
                else "没有找到足够证据，暂时无法回答。"
            )
            status = "answered" if case.should_answer else "insufficient_evidence"
            return PipelineResult(
                answer=answer,
                evidence=evidence,
                validation=ValidationResult(
                    is_valid=True,
                    citations=tuple(case.expected_doc_ids),
                    reason="通过",
                ),
                trace=PipelineTrace(
                    strategy="hybrid",
                    retry_count=0,
                    retrieval_calls=1,
                    generation_calls=int(case.should_answer),
                    latency_ms=12.5,
                    estimated_input_tokens=30,
                    estimated_output_tokens=10,
                ),
                status=status,
            )

        report = evaluate_cases(cases, run_case)

        for metric in (
            "hit_at_1",
            "hit_at_3",
            "mrr",
            "context_precision",
            "context_recall",
            "answer_accuracy",
            "refusal_accuracy",
            "latency_p50_ms",
            "latency_p95_ms",
            "avg_retrieval_calls",
            "avg_generation_calls",
            "estimated_tokens",
        ):
            self.assertIn(metric, report.metrics)

    def test_markdown_report_contains_metric_definitions_and_failures(self):
        cases = load_evaluation_cases(CASES_PATH)[:2]
        report = evaluate_cases(
            cases,
            lambda case: PipelineResult(
                answer="没有找到足够证据，暂时无法回答。",
                evidence=[],
                validation=ValidationResult(False, (), "没有证据"),
                trace=PipelineTrace("hybrid", 1, 2, 0, 8.0, 20, 0),
                status="insufficient_evidence",
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "report.md"
            write_markdown_report(report, target)
            content = target.read_text(encoding="utf-8")

        self.assertIn("指标口径", content)
        self.assertIn("失败案例", content)
        self.assertIn("Hit@1", content)


if __name__ == "__main__":
    unittest.main()
