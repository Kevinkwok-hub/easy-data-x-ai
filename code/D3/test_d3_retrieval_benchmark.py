import tempfile
import unittest
from pathlib import Path

from d3_5_evaluate import hybrid_retrieve, vector_retrieve
from d3_6_benchmark import build_seekdb_retrievers
from rag_engineering import analyze_query
from rag_evaluation import CASES_PATH, load_evaluation_cases
from retrieval_benchmark import (
    BenchmarkConfig,
    CostAssumptions,
    render_markdown,
    run_retrieval_triangle_benchmark,
    write_markdown_report,
)


class RetrievalTriangleBenchmarkTests(unittest.TestCase):
    def test_rejects_invalid_benchmark_and_cost_settings(self):
        with self.assertRaisesRegex(ValueError, "measurement_rounds"):
            BenchmarkConfig(measurement_rounds=0)
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            CostAssumptions(input_price_per_million_tokens=-1)

    def test_compares_answerable_cases_with_reproducible_quality_metrics(self):
        report = run_retrieval_triangle_benchmark(
            load_evaluation_cases(CASES_PATH),
            vector_retriever=vector_retrieve,
            hybrid_retriever=hybrid_retrieve,
            config=BenchmarkConfig(warmup_rounds=0, measurement_rounds=1),
        )
        by_name = {result.strategy: result for result in report.strategies}

        self.assertEqual(50, report.case_count)
        self.assertEqual(50, report.samples_per_strategy)
        self.assertEqual(0.72, by_name["vector"].hit_at_1)
        self.assertEqual(0.88, by_name["hybrid"].hit_at_1)
        self.assertGreater(
            by_name["hybrid"].average_context_tokens,
            by_name["vector"].average_context_tokens,
        )
        self.assertEqual(0, by_name["vector"].fulltext_branches_per_query)
        self.assertEqual(1, by_name["hybrid"].fulltext_branches_per_query)

    def test_cost_formula_uses_replaceable_input_token_price(self):
        report = run_retrieval_triangle_benchmark(
            load_evaluation_cases(CASES_PATH),
            vector_retriever=vector_retrieve,
            hybrid_retriever=hybrid_retrieve,
            config=BenchmarkConfig(
                warmup_rounds=0,
                measurement_rounds=1,
                cost=CostAssumptions(
                    input_price_per_million_tokens=2.0,
                    currency="TEST",
                ),
            ),
        )
        vector = next(
            result for result in report.strategies if result.strategy == "vector"
        )

        self.assertAlmostEqual(
            vector.average_context_tokens * 2.0 / 1_000,
            vector.context_cost_per_1k_queries,
            places=4,
        )

    def test_markdown_explains_scope_formula_and_tradeoff(self):
        report = run_retrieval_triangle_benchmark(
            load_evaluation_cases(CASES_PATH),
            vector_retriever=vector_retrieve,
            hybrid_retriever=hybrid_retrieve,
            config=BenchmarkConfig(warmup_rounds=0, measurement_rounds=1),
        )
        content = render_markdown(report)

        self.assertIn("延迟 / 成本 / 精度三角实验", content)
        self.assertIn("只包围检索函数", content)
        self.assertIn("千次查询上下文成本", content)
        self.assertIn("自适应路由", content)
        self.assertIn("不应被当作", content)
        self.assertIn("不会真的发出数据库", content)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "triangle.md"
            write_markdown_report(report, target)
            self.assertEqual(content, target.read_text(encoding="utf-8"))

    def test_seekdb_adapter_uses_one_filtered_hybrid_request(self):
        class FakeCollection:
            def __init__(self):
                self.vector_calls = []
                self.hybrid_calls = []

            def query(self, **kwargs):
                self.vector_calls.append(kwargs)
                return {
                    "ids": [["kb_001"]],
                    "documents": [["OB-4.2.1 版本兼容性说明"]],
                    "distances": [[0.1]],
                    "metadatas": [[{"version": "4.2.1"}]],
                }

            def hybrid_search(self, **kwargs):
                self.hybrid_calls.append(kwargs)
                return {
                    "ids": [["kb_001"]],
                    "documents": [["OB-4.2.1 版本兼容性说明"]],
                    "distances": [[0.9]],
                    "metadatas": [[{"version": "4.2.1"}]],
                }

        collection = FakeCollection()
        vector, hybrid = build_seekdb_retrievers(collection)
        question = "OB-4.2.1 和旧版本兼容吗？"
        analysis = analyze_query(question)

        vector(question, analysis)
        groups = hybrid(question, analysis)

        self.assertEqual(1, len(collection.vector_calls))
        self.assertEqual(1, len(collection.hybrid_calls))
        call = collection.hybrid_calls[0]
        self.assertEqual({"version": "4.2.1"}, call["query"]["where"])
        self.assertEqual({"version": "4.2.1"}, call["knn"]["where"])
        self.assertEqual("kb_001", groups[0][0].doc_id)


if __name__ == "__main__":
    unittest.main()
