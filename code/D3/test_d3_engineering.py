import unittest

from rag_engineering import (
    Evidence,
    analyze_query,
    build_context,
    fuse_and_rerank,
    run_engineering_pipeline,
    validate_answer,
)


class QueryAnalysisTests(unittest.TestCase):
    def test_extracts_exact_identifiers_and_routes_to_hybrid_search(self):
        analysis = analyze_query("OB-4.2.1 遇到 E-4012 时该怎么处理？")

        self.assertEqual("hybrid", analysis.strategy)
        self.assertEqual("OB-4.2.1", analysis.filters["version"])
        self.assertIn("E-4012", analysis.keywords)

    def test_routes_general_explanation_to_vector_search(self):
        analysis = analyze_query("怎么优化数据库的查询性能？")

        self.assertEqual("vector", analysis.strategy)
        self.assertEqual({}, analysis.filters)


class RetrievalFusionTests(unittest.TestCase):
    def test_filters_unauthorized_evidence_before_ranking(self):
        ranked = fuse_and_rerank(
            [
                [
                    Evidence("public", "公开手册", 0.8, "vector", access_level="public"),
                    Evidence("secret", "内部财务预测", 0.99, "vector", access_level="internal"),
                ],
                [
                    Evidence("secret", "内部财务预测", 0.99, "keyword", access_level="internal"),
                    Evidence("public", "公开手册", 0.7, "keyword", access_level="public"),
                ],
            ],
            allowed_access_levels={"public"},
        )

        self.assertEqual(["public"], [item.doc_id for item in ranked])

    def test_rrf_deduplicates_documents_and_preserves_multiple_sources(self):
        ranked = fuse_and_rerank(
            [
                [
                    Evidence("kb_004", "连接池耗尽", 0.9, "vector"),
                    Evidence("kb_005", "认证超时", 0.8, "vector"),
                ],
                [
                    Evidence("kb_004", "连接池耗尽", 1.0, "keyword"),
                    Evidence("kb_006", "SQL 解析失败", 0.7, "keyword"),
                ],
            ]
        )

        self.assertEqual("kb_004", ranked[0].doc_id)
        self.assertEqual({"vector", "keyword"}, set(ranked[0].sources))
        self.assertEqual(3, len(ranked))

    def test_fresh_trusted_document_wins_when_relevance_is_tied(self):
        ranked = fuse_and_rerank(
            [
                [
                    Evidence(
                        "old",
                        "旧说明",
                        0.8,
                        "vector",
                        metadata={"updated_at": "2025-01-01", "source_rank": "1"},
                    )
                ],
                [
                    Evidence(
                        "current",
                        "当前说明",
                        0.8,
                        "keyword",
                        metadata={"updated_at": "2026-06-01", "source_rank": "2"},
                    )
                ],
            ]
        )

        self.assertEqual("current", ranked[0].doc_id)

    def test_context_respects_budget_and_keeps_evidence_ids(self):
        context = build_context(
            [
                Evidence("kb_001", "甲" * 20, 1.0, "vector"),
                Evidence("kb_002", "乙" * 20, 0.9, "vector"),
            ],
            max_chars=38,
        )

        self.assertLessEqual(len(context), 38)
        self.assertIn("[kb_001]", context)
        self.assertNotIn("[kb_002]", context)

    def test_context_skips_oversized_evidence_and_keeps_shorter_evidence(self):
        context = build_context(
            [
                Evidence("too-long", "甲" * 100, 1.0, "vector"),
                Evidence("short", "可用短证据", 0.9, "keyword"),
            ],
            max_chars=30,
        )

        self.assertLessEqual(len(context), 30)
        self.assertNotIn("[too-long]", context)
        self.assertIn("[short]", context)


class AnswerValidationTests(unittest.TestCase):
    def test_accepts_answer_supported_by_known_citation(self):
        result = validate_answer(
            "连接池耗尽时应检查连接泄漏。[kb_004]",
            [Evidence("kb_004", "连接池耗尽时应检查连接泄漏。", 1.0, "keyword")],
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(("kb_004",), result.citations)

    def test_rejects_missing_or_unknown_citations(self):
        evidence = [Evidence("kb_004", "连接池耗尽", 1.0, "keyword")]

        self.assertFalse(validate_answer("建议直接重启。", evidence).is_valid)
        self.assertFalse(validate_answer("建议直接重启。[kb_999]", evidence).is_valid)

    def test_pipeline_rewrites_and_retries_once_after_weak_evidence(self):
        queries = []

        def retrieve(query, analysis):
            queries.append(query)
            if len(queries) == 1:
                return []
            return [[Evidence("kb_004", "E-4012 表示连接池耗尽。", 1.0, "keyword")]]

        result = run_engineering_pipeline(
            "E-4012 怎么办？",
            retrieve_fn=retrieve,
            generate_fn=lambda question, context: "E-4012 表示连接池耗尽。[kb_004]",
            max_retries=1,
        )

        self.assertEqual(2, len(queries))
        self.assertTrue(result.validation.is_valid)
        self.assertEqual(1, result.trace.retry_count)
        self.assertEqual("answered", result.status)

    def test_pipeline_refuses_when_retries_still_have_no_evidence(self):
        result = run_engineering_pipeline(
            "尚未收录的功能什么时候发布？",
            retrieve_fn=lambda query, analysis: [],
            generate_fn=lambda question, context: "不应调用",
            max_retries=1,
        )

        self.assertEqual("insufficient_evidence", result.status)
        self.assertIn("没有找到足够证据", result.answer)

    def test_pipeline_distinguishes_access_denied_from_missing_evidence(self):
        result = run_engineering_pipeline(
            "下一季度内部预测是多少？",
            retrieve_fn=lambda query, analysis: [
                Evidence(
                    "internal-forecast",
                    "内部预测",
                    1.0,
                    "keyword",
                    access_level="internal",
                )
            ],
            generate_fn=lambda question, context: "不应调用",
            allowed_access_levels={"public"},
            max_retries=0,
        )

        self.assertEqual("access_denied", result.status)
        self.assertIn("无权访问", result.answer)
        self.assertNotIn("internal-forecast", result.answer)


if __name__ == "__main__":
    unittest.main()
