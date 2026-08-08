import unittest

from rag_engineering import (
    Evidence,
    adaptive_retrieve,
    analyze_query,
    build_context,
    build_retrieval_plan,
    fuse_and_rerank,
    grade_evidence,
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

    def test_rewrites_colloquial_query_and_keeps_exact_identifiers(self):
        analysis = analyze_query("OB-4.2.1 连不上咋办？")

        self.assertEqual("OB-4.2.1 连接失败如何处理？", analysis.rewritten_query)
        self.assertEqual("procedure", analysis.intent)
        self.assertEqual("OB-4.2.1", analysis.filters["version"])
        self.assertEqual(("keyword", "vector"), analysis.routes)

        error_analysis = analyze_query("报错 4012 怎么处理？")
        self.assertIn("E-4012", error_analysis.rewritten_query)
        self.assertEqual("E-4012", error_analysis.filters["error_code"])

    def test_decomposes_parallel_question_and_adds_structured_route(self):
        analysis = analyze_query(
            "查询 OB-4.2.1 与 OB-4.1.0 当前状态，同时给出 E-4012 的排查步骤"
        )

        self.assertEqual(2, len(analysis.sub_queries))
        self.assertEqual("OB-4.2.1,OB-4.1.0", analysis.filters["versions"])
        self.assertEqual("E-4012", analysis.filters["error_code"])
        self.assertEqual(
            ("structured", "keyword", "vector"),
            analysis.routes,
        )


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_executes_query_plan_across_routes_and_corrective_fallback(self):
        calls = []

        def make_retriever(route):
            def retrieve(query, analysis):
                calls.append((route, query))
                return [Evidence(f"{route}-{len(calls)}", query, 1.0, route)]

            return retrieve

        analysis = analyze_query("OB-4.2.1 遇到 E-4012 如何处理？")
        result = adaptive_retrieve(
            analysis,
            retrievers={
                route: make_retriever(route)
                for route in ("keyword", "vector", "fallback")
            },
            retry_count=1,
        )

        self.assertEqual(("keyword", "vector", "fallback"), result.plan.routes)
        self.assertEqual(2, len(result.plan.queries))
        self.assertEqual(6, result.calls)
        self.assertEqual(6, len(result.candidate_groups))
        self.assertIn("官方文档", result.plan.queries[-1])

    def test_plan_rejects_missing_matching_retriever(self):
        analysis = analyze_query("OB-4.2.1 的兼容性如何？")

        with self.assertRaisesRegex(ValueError, "没有与查询计划匹配"):
            build_retrieval_plan(
                analysis,
                available_routes={"structured"},
            )

    def test_grades_and_filters_weak_evidence_before_generation(self):
        evidence = [
            Evidence("strong", "相关证据", 1.0, "vector"),
            Evidence("weak", "无关内容", 0.9, "vector"),
        ]

        accepted, grades = grade_evidence(
            "问题",
            evidence,
            grader_fn=lambda question, item: 0.9 if item.doc_id == "strong" else 0.2,
            threshold=0.5,
        )

        self.assertEqual(["strong"], [item.doc_id for item in accepted])
        self.assertEqual(("strong", 0.9), grades[0])


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
        self.assertFalse(validate_answer("[kb_004]", evidence).is_valid)

    def test_rejects_an_uncited_claim_even_when_another_claim_is_cited(self):
        result = validate_answer(
            "应先检查连接泄漏。[kb_004] 然后直接扩容。",
            [Evidence("kb_004", "应先检查连接泄漏。", 1.0, "keyword")],
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(("然后直接扩容。",), result.unsupported_claims)

    def test_does_not_treat_arbitrary_colon_line_as_a_safe_heading(self):
        result = validate_answer(
            "数据库已经恢复：\n处理完成。[kb_004]",
            [Evidence("kb_004", "处理完成。", 1.0, "keyword")],
        )

        self.assertFalse(result.is_valid)
        self.assertIn("数据库已经恢复", result.unsupported_claims[0])

    def test_accepts_injected_semantic_support_check(self):
        evidence = [Evidence("kb_004", "应先检查连接泄漏。", 1.0, "keyword")]

        result = validate_answer(
            "建议立即重启。[kb_004]",
            evidence,
            question="E-4012 怎么办？",
            support_fn=lambda question, claim, cited: "重启" not in claim,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("建议立即重启", result.unsupported_claims[0])

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
        self.assertIn("相关证据无访问权限", result.trace.validation_failures)

    def test_pipeline_records_adaptive_routes_grades_and_retry_queries(self):
        def weak_vector(query, analysis):
            return [Evidence("weak", "无关的营销介绍", 0.9, "vector")]

        def exact_keyword(query, analysis):
            return []

        def corrective_fallback(query, analysis):
            if "官方文档" not in query:
                return []
            return [Evidence("kb_004", "E-4012 表示连接池耗尽。", 1.0, "fallback")]

        result = run_engineering_pipeline(
            "E-4012 如何处理？",
            route_retrievers={
                "vector": weak_vector,
                "keyword": exact_keyword,
                "fallback": corrective_fallback,
            },
            evidence_grader_fn=(
                lambda question, item: 0.9 if item.doc_id == "kb_004" else 0.1
            ),
            generate_fn=(
                lambda question, context: "E-4012 表示连接池耗尽。[kb_004]"
            ),
            max_retries=1,
        )

        self.assertEqual("answered", result.status)
        self.assertEqual(("keyword", "vector", "fallback"), result.trace.retrieval_routes)
        self.assertIn("官方文档", result.trace.query_history[-1])
        self.assertIn(("weak", 0.1), result.trace.evidence_grades)
        self.assertIn(("kb_004", 0.9), result.trace.evidence_grades)
        self.assertIn("检索结果未通过相关性评分", result.trace.validation_failures)


if __name__ == "__main__":
    unittest.main()
