from __future__ import annotations

import unittest

from run_tests import extract_test_count


class TestRunnerSafetyTests(unittest.TestCase):
    def test_extracts_positive_test_count(self) -> None:
        self.assertEqual(extract_test_count("Ran 12 tests in 0.2s"), 12)

    def test_rejects_zero_or_missing_test_count(self) -> None:
        for output in ("Ran 0 tests in 0.0s", "unexpected output"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, "测试"):
                    extract_test_count(output)

    def test_rejects_ambiguous_or_zero_test_summaries(self) -> None:
        for output in (
            "Ran 1 test\nRan 0 tests\nOK",
            "Ran 2 tests\nRan 2 tests\nOK",
        ):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, "测试"):
                    extract_test_count(output)


if __name__ == "__main__":
    unittest.main()
