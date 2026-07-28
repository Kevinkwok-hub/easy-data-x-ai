import importlib
import math
import unittest
from unittest.mock import patch

from D2.d2_chunking_strategies import (
    _detect_boundary_positions,
    dynamic_overlap_chunk,
    fixed_overlap_chunk,
    parent_child_chunk,
    semantic_chunk,
)


class FixedOverlapValidationTests(unittest.TestCase):
    def test_rejects_non_positive_size_and_invalid_overlap(self):
        invalid_parameters = (
            {"chunk_size": 0, "overlap": 0},
            {"chunk_size": -1, "overlap": 0},
            {"chunk_size": 3, "overlap": -1},
            {"chunk_size": 3, "overlap": 3},
            {"chunk_size": 3, "overlap": 4},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    fixed_overlap_chunk("", **parameters)


class SemanticChunkingTests(unittest.TestCase):
    def test_rejects_non_numeric_non_finite_or_out_of_range_percentile(self):
        text = "第一句。第二句。"
        invalid_values = (None, "95", math.nan, math.inf, -1, 101)

        for percentile in invalid_values:
            with self.subTest(percentile=percentile):
                with self.assertRaisesRegex(ValueError, "percentile"):
                    semantic_chunk(
                        text,
                        lambda _sentences: [[1.0, 0.0], [0.0, 1.0]],
                        percentile=percentile,
                        min_chunk_chars=1,
                        max_chunk_chars=100,
                    )

    def test_identical_embeddings_do_not_create_breakpoints(self):
        text = "连接池需要合理配置。连接池需要持续监控。连接池需要避免泄漏。"

        def embed_same(sentences):
            return [[1.0, 0.0] for _ in sentences]

        chunks = semantic_chunk(
            text,
            embed_same,
            min_chunk_chars=1,
            max_chunk_chars=1000,
        )

        self.assertEqual(chunks, [text])

    def test_rejects_invalid_minimum_and_maximum_lengths(self):
        invalid_parameters = (
            {"min_chunk_chars": 0, "max_chunk_chars": 10},
            {"min_chunk_chars": 1, "max_chunk_chars": 0},
            {"min_chunk_chars": 11, "max_chunk_chars": 10},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    semantic_chunk("", lambda _sentences: [], **parameters)

    def test_rejects_embedding_count_mismatch(self):
        text = "第一句。第二句。第三句。"

        for embeddings in (
            [[1.0, 0.0]],
            [[1.0, 0.0]] * 4,
        ):
            with self.subTest(embedding_count=len(embeddings)):
                with self.assertRaisesRegex(ValueError, "数量"):
                    semantic_chunk(text, lambda _sentences, result=embeddings: result)

    def test_rejects_empty_mismatched_and_non_finite_embeddings(self):
        text = "第一句。第二句。"
        invalid_embeddings = (
            [[], []],
            [[1.0, 0.0], [1.0]],
            [[1.0, math.nan], [1.0, 0.0]],
            [[1.0, math.inf], [1.0, 0.0]],
        )

        for embeddings in invalid_embeddings:
            with self.subTest(embeddings=embeddings):
                with self.assertRaisesRegex(ValueError, "向量"):
                    semantic_chunk(text, lambda _sentences, result=embeddings: result)

    def test_never_returns_chunks_larger_than_maximum(self):
        text = "甲" * 250

        chunks = semantic_chunk(
            text,
            lambda _sentences: [],
            min_chunk_chars=1,
            max_chunk_chars=100,
        )

        self.assertEqual(chunks, ["甲" * 100, "甲" * 100, "甲" * 50])


class DynamicOverlapTests(unittest.TestCase):
    def test_detects_markdown_structure_boundaries(self):
        text = (
            "# 标题\n"
            "正文段落\n\n"
            "| 字段 | 含义 |\n"
            "| --- | --- |\n"
            "| code | 错误码 |\n\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
        )

        boundaries = _detect_boundary_positions(text)

        self.assertIn(text.index("| 字段 | 含义 |"), boundaries)
        self.assertIn(text.index("```python"), boundaries)
        self.assertIn(text.rindex("```"), boundaries)

    def test_rejects_invalid_sizes_and_overlaps(self):
        invalid_parameters = (
            {"chunk_size": 0, "base_overlap": 0, "max_overlap": 0},
            {"chunk_size": 10, "base_overlap": -1, "max_overlap": 2},
            {"chunk_size": 10, "base_overlap": 3, "max_overlap": 2},
            {"chunk_size": 10, "base_overlap": 2, "max_overlap": 10},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    dynamic_overlap_chunk("", **parameters)


class ParentChildValidationTests(unittest.TestCase):
    def test_rejects_invalid_sizes_and_child_overlap(self):
        invalid_parameters = (
            {"parent_size": 0, "child_size": 10, "child_overlap": 0},
            {"parent_size": 10, "child_size": 0, "child_overlap": 0},
            {"parent_size": 10, "child_size": 5, "child_overlap": -1},
            {"parent_size": 10, "child_size": 5, "child_overlap": 5},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    parent_child_chunk("", **parameters)


class CompareScriptTests(unittest.TestCase):
    def test_compare_script_can_be_imported_as_module(self):
        module = importlib.import_module("D2.d2_5_chunking_compare")

        self.assertTrue(hasattr(module, "main"))

    def test_blank_and_template_api_keys_are_not_available(self):
        module = importlib.import_module("D2.d2_5_chunking_compare")

        for api_key in (
            "   ",
            "  YOUR_API_KEY  ",
            "  your_siliconflow_api_key_here  ",
        ):
            with self.subTest(api_key=api_key):
                with patch.object(module.Config, "SILICONFLOW_API_KEY", api_key):
                    self.assertFalse(module.has_siliconflow_key())


if __name__ == "__main__":
    unittest.main()
