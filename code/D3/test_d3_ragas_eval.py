"""D3 RAGAS 评测集、金标指标与配置的单元测试（不调用外部 API）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from d3_5_ragas_eval import (
    build_summary,
    ensure_complete_metric_scores,
    ensure_no_pipeline_failures,
    hybrid_search,
    run_evaluation,
    run_ragas,
    write_run_artifacts,
)
from d3_eval_core import (
    aggregate_rows,
    attach_ragas_scores,
    compute_retrieval_metrics,
    extract_lexical_anchor,
    extract_metadata_filter,
    load_json,
    paired_metric_comparison,
    sha256_file,
    validate_dataset,
)
from rag_data import knowledge_chunks

DATA_DIR = Path(__file__).resolve().parent / "data"


class DatasetSchemaTests(unittest.TestCase):
    def test_course_dataset_has_four_balanced_scenarios(self):
        dataset = load_json(DATA_DIR / "eval_dataset.json")
        known_ids = {chunk["id"] for chunk in knowledge_chunks}
        counts = {}
        for case in dataset["cases"]:
            scenario = case["scenario"]["type"]
            counts[scenario] = counts.get(scenario, 0) + 1

        self.assertEqual(len(dataset["cases"]), 20)
        self.assertEqual(counts, {"exact": 5, "multi_hop": 5, "temporal": 5, "fuzzy": 5})
        self.assertEqual(validate_dataset(dataset, known_ids=known_ids), [])
        for case in dataset["cases"]:
            self.assertTrue(
                extract_lexical_anchor(case["question"]),
                msg=f"{case['id']} 缺少独立全文查询",
            )

    def test_gold_references_are_closed_without_forbidden_field(self):
        dataset = load_json(DATA_DIR / "eval_dataset.json")
        chunks = {chunk["id"]: chunk["content"] for chunk in knowledge_chunks}

        for case in dataset["cases"]:
            self.assertNotIn("forbidden_chunk_ids", case.get("gold", {}))
            for fact in case["gold"]["required_facts"]:
                for chunk_id in fact["acceptable_chunk_ids"]:
                    self.assertIn(chunk_id, chunks)

        fuzzy_01 = next(case for case in dataset["cases"] if case["id"] == "fuzzy_01")
        self.assertEqual(
            [fact["fact_id"] for fact in fuzzy_01["gold"]["required_facts"]],
            ["faq_intent"],
        )
        fuzzy_05 = next(case for case in dataset["cases"] if case["id"] == "fuzzy_05")
        self.assertNotIn("澄清", fuzzy_05["reference"])

    def test_validate_dataset_rejects_missing_collection_ids(self):
        dataset = load_json(DATA_DIR / "eval_dataset.json")
        errors = validate_dataset(dataset, known_ids={"kb_001"}, top_k=4)
        self.assertTrue(any("金标 chunk 不存在" in item for item in errors))

    def test_validate_dataset_rejects_top_k_too_small_for_multi_hop(self):
        dataset = load_json(DATA_DIR / "eval_dataset.json")
        known_ids = {chunk["id"] for chunk in knowledge_chunks}
        errors = validate_dataset(dataset, known_ids=known_ids, top_k=1)
        self.assertTrue(any("必需事实数超过 top_k=1" in item for item in errors))


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_json(DATA_DIR / "eval_dataset.json")
        self.multi_case = next(
            case for case in self.dataset["cases"] if case["id"] == "multi_01"
        )
        self.temporal_case = next(
            case for case in self.dataset["cases"] if case["id"] == "temporal_01"
        )

    def test_multi_hop_requires_all_reference_facts(self):
        partial = compute_retrieval_metrics(self.multi_case, ["kb_004"])
        complete = compute_retrieval_metrics(self.multi_case, ["kb_004", "kb_005"])

        self.assertEqual(partial["top1_reference_hit"], 1.0)
        self.assertEqual(partial["all_required_evidence_at_k"], 0.0)
        self.assertEqual(complete["all_required_evidence_at_k"], 1.0)

    def test_temporal_metric_records_stale_evidence(self):
        metrics = compute_retrieval_metrics(
            self.temporal_case, ["kb_007", "kb_008", "kb_013"]
        )

        self.assertEqual(metrics["gold_chunk_recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["stale_evidence_rate"], 2 / 3)

    def test_anchor_is_derived_from_visible_identifiers_only(self):
        self.assertEqual(extract_lexical_anchor("E-4012 怎么解决？"), "E-4012")
        self.assertEqual(
            extract_lexical_anchor("DBMS_HYBRID_SEARCH 怎么用？"),
            "DBMS_HYBRID_SEARCH",
        )
        self.assertEqual(
            extract_lexical_anchor("连不上数据库怎么办？"),
            "连接",
        )
        self.assertEqual(
            extract_lexical_anchor("parallel_degree 参数怎么配置？"),
            "parallel_degree",
        )
        self.assertEqual(extract_lexical_anchor("完全未知的问法"), "")

    def test_single_version_uses_metadata_filter(self):
        self.assertEqual(
            extract_metadata_filter("OB-4.2.1 版本兼容吗？"),
            {"version": "4.2.1"},
        )
        self.assertEqual(
            extract_metadata_filter("从 OB-3.2.4 升级到 OB-4.2.1"),
            {},
        )

    def test_hybrid_search_uses_short_query_and_filters_both_branches(self):
        class RecordingCollection:
            def __init__(self):
                self.call = None

            def hybrid_search(self, **kwargs):
                self.call = kwargs
                return {"documents": [["doc"]], "ids": [["kb_001"]]}

        collection = RecordingCollection()
        documents, ids = hybrid_search(
            collection,
            "OB-4.2.1 版本和旧版本兼容吗？",
            4,
        )

        self.assertEqual((documents, ids), (["doc"], ["kb_001"]))
        self.assertEqual(
            collection.call["query"]["where_document"]["$contains"],
            "兼容",
        )
        self.assertEqual(collection.call["query"]["where"], {"version": "4.2.1"})
        self.assertEqual(collection.call["knn"]["where"], {"version": "4.2.1"})
        self.assertEqual(collection.call["query"]["n_results"], 6)


class AggregationTests(unittest.TestCase):
    def test_retrieval_only_run_cannot_publish_ragas_summary(self):
        args = SimpleNamespace(mode="retrieval", publish_summary=Path("summary.json"))

        with self.assertRaisesRegex(ValueError, "仅支持 --mode ragas"):
            run_evaluation(args)

    def test_ragas_scores_attach_by_input_order(self):
        rows = [{"case_id": "a"}, {"case_id": "b"}]
        scores = [{"faithfulness": 0.8}, {"faithfulness": float("nan")}]

        attach_ragas_scores(rows, scores)

        self.assertEqual(rows[0]["ragas_scores"]["faithfulness"], 0.8)
        self.assertIsNone(rows[1]["ragas_scores"]["faithfulness"])

    def test_summary_keeps_ragas_and_retrieval_metrics_separate(self):
        vector_rows = [self._row("case_01", "exact", 0.0, 0.3)]
        hybrid_rows = [self._row("case_01", "exact", 1.0, 0.8)]
        dataset = {"dataset_id": "test-dataset", "cases": [{}]}

        summary = build_summary(dataset, "ragas", 4, vector_rows, hybrid_rows)

        vector = summary["strategies"]["vector"]["overall"]["metrics"]
        hybrid = summary["strategies"]["hybrid"]["overall"]["metrics"]
        self.assertEqual(vector["top1_reference_hit"], 0.0)
        self.assertEqual(hybrid["top1_reference_hit"], 1.0)
        self.assertEqual(vector["faithfulness"], 0.3)
        self.assertEqual(hybrid["faithfulness"], 0.8)

    def test_aggregation_counts_only_valid_scores(self):
        rows = [
            {"retrieval_metrics": {"top1_reference_hit": 1.0}},
            {"retrieval_metrics": {"top1_reference_hit": 0.0}},
            {"retrieval_metrics": {"top1_reference_hit": None}},
        ]

        result = aggregate_rows(rows, ["top1_reference_hit"])

        self.assertEqual(result["n_total"], 3)
        self.assertEqual(result["n_scored"]["top1_reference_hit"], 2)
        self.assertEqual(result["metrics"]["top1_reference_hit"], 0.5)

    def test_paired_comparison_reports_effect_interval_and_significance(self):
        vector_rows = [
            self._row(f"case_{index}", "exact", 0.0, 0.2)
            for index in range(20)
        ]
        hybrid_rows = [
            self._row(f"case_{index}", "exact", 1.0, 0.8)
            for index in range(20)
        ]

        result = paired_metric_comparison(
            vector_rows,
            hybrid_rows,
            "faithfulness",
            resamples=1_000,
        )

        self.assertEqual(result["n_pairs"], 20)
        self.assertAlmostEqual(result["mean_delta"], 0.6)
        self.assertEqual(result["ci95"], [0.6, 0.6])
        self.assertTrue(result["significant_at_0_05"])

    def test_pipeline_failures_and_missing_scores_block_publication(self):
        with self.assertRaisesRegex(RuntimeError, "vector/case_01"):
            ensure_no_pipeline_failures(
                {
                    "vector": [
                        {
                            "case_id": "case_01",
                            "failure": {"stage": "retrieval"},
                        }
                    ]
                }
            )
        with self.assertRaisesRegex(RuntimeError, "case_01/context_recall"):
            ensure_complete_metric_scores(
                [{"case_id": "case_01", "ragas_scores": {}}]
            )

    def test_ragas_raises_metric_errors_instead_of_dropping_scores(self):
        class FakeDataset:
            @staticmethod
            def from_list(rows):
                return rows

        observed = {}

        def fake_evaluate(dataset, **kwargs):
            observed["dataset"] = dataset
            observed.update(kwargs)
            return SimpleNamespace(
                scores=[
                    {
                        "context_recall": 0.8,
                        "context_precision": 0.7,
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.6,
                    }
                ]
            )

        fake_modules = {
            "datasets": SimpleNamespace(Dataset=FakeDataset),
            "ragas": SimpleNamespace(evaluate=fake_evaluate),
            "ragas.metrics": SimpleNamespace(
                context_recall=object(),
                context_precision=object(),
                faithfulness=object(),
                answer_relevancy=object(),
            ),
        }
        rows = [
            {
                "case_id": "case_01",
                "question": "question",
                "answer": "answer",
                "retrieved_contexts": ["context"],
                "reference": "reference",
            }
        ]
        with patch.dict(sys.modules, fake_modules), patch(
            "d3_5_ragas_eval.build_ragas_dependencies",
            return_value=("llm", "embeddings"),
        ):
            run_ragas(rows)

        self.assertTrue(observed["raise_exceptions"])
        self.assertEqual(rows[0]["ragas_scores"]["faithfulness"], 0.9)

    def test_run_artifacts_include_manifest_and_case_rows_without_keys(self):
        vector_rows = [self._row("case_01", "exact", 0.0, 0.3)]
        hybrid_rows = [self._row("case_01", "exact", 1.0, 0.8)]
        dataset = {"dataset_id": "test-dataset", "cases": [{}]}
        summary = build_summary(dataset, "retrieval", 4, vector_rows, hybrid_rows)
        args = SimpleNamespace(
            collection="d3_product_kb",
            mode="retrieval",
            publish_summary=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            dataset_path = run_dir / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            args.publish_summary = run_dir / "published.json"
            write_run_artifacts(
                run_dir,
                dataset_path,
                "fingerprint",
                19,
                summary,
                vector_rows,
                hybrid_rows,
                args,
                None,
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["knowledge_base"]["document_count"], 19)
            self.assertTrue((run_dir / "case_results.vector.jsonl").exists())
            self.assertTrue((run_dir / "case_results.hybrid.jsonl").exists())
            published = json.loads(
                (run_dir / "published.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published["dataset"]["sha256"], sha256_file(dataset_path))
            self.assertNotIn("path", published["dataset"])
            self.assertEqual(
                published["knowledge_base"]["fingerprint"],
                "fingerprint",
            )
            self.assertIn("paired_comparison", published)
            self.assertEqual(len(published["case_results"]["vector"]), 1)
            self.assertEqual(len(published["case_results"]["hybrid"]), 1)
            text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-", text)

    @staticmethod
    def _row(case_id, scenario, top1_hit, faithfulness):
        return {
            "case_id": case_id,
            "scenario": scenario,
            "retrieval_metrics": {
                "gold_chunk_recall_at_k": top1_hit,
                "gold_chunk_precision_at_k": top1_hit,
                "top1_reference_hit": top1_hit,
                "evidence_coverage_at_k": top1_hit,
                "all_required_evidence_at_k": top1_hit,
                "stale_evidence_rate": 0.0,
            },
            "ragas_scores": {
                "context_recall": faithfulness,
                "context_precision": faithfulness,
                "faithfulness": faithfulness,
                "answer_relevancy": faithfulness,
            },
        }


class RagasConfigurationTests(unittest.TestCase):
    def test_ragas_requires_explicit_embedding_model(self):
        with patch.object(Config, "RAGAS_API_KEY", "sk-example"), patch.object(
            Config, "RAGAS_EMBEDDING_MODEL", ""
        ):
            config = Config.get_ragas_evaluator_config()

        self.assertEqual(config["missing"], ["RAGAS_EMBEDDING_MODEL"])

    def test_ragas_falls_back_to_siliconflow_key(self):
        with patch.object(Config, "RAGAS_API_KEY", ""), patch.object(
            Config, "SILICONFLOW_API_KEY", "sk-silicon"
        ), patch.object(Config, "RAGAS_EMBEDDING_MODEL", "BAAI/bge-m3"):
            config = Config.get_ragas_evaluator_config()

        self.assertEqual(config["api_key"], "sk-silicon")
        self.assertEqual(config["missing"], [])

    def test_generation_client_uses_ragas_config(self):
        from d3_5_ragas_eval import build_generation_client

        with patch.object(Config, "RAGAS_API_KEY", "sk-ragas"), patch.object(
            Config, "RAGAS_BASE_URL", "https://example.test/v1"
        ), patch.object(Config, "RAGAS_LLM_MODEL", "demo-model"), patch.object(
            Config, "RAGAS_EMBEDDING_MODEL", "BAAI/bge-m3"
        ), patch("openai.OpenAI") as client_cls:
            client, model = build_generation_client()

        client_cls.assert_called_once_with(
            api_key="sk-ragas",
            base_url="https://example.test/v1",
        )
        self.assertEqual(model, "demo-model")
        self.assertIs(client, client_cls.return_value)


if __name__ == "__main__":
    unittest.main()
