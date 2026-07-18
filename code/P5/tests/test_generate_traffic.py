from __future__ import annotations

import unittest
from unittest.mock import patch

from app.observability import generate_traffic
from app.observability.generate_traffic import TrafficError


class GenerateTrafficTest(unittest.TestCase):
    def test_generate_traffic_cycles_required_case_types(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_post(url: str, payload: dict[str, object], *, timeout_seconds: float) -> dict[str, object]:
            calls.append((url, payload))
            return {
                "task_id": payload["task_id"],
                "answer": "包含预期内容：" + str(payload["expected_answer_contains"]),
                "task_success": True,
                "handoff": False,
            }

        with patch.object(generate_traffic, "_post_json", side_effect=fake_post):
            summary = generate_traffic.generate_traffic(count=7, base_url="http://agent.local/")

        self.assertEqual(summary.total_sent, 7)
        self.assertEqual(
            summary.by_kind,
            {
                "success": 1,
                "retrieval_miss": 1,
                "retrieval_failure": 1,
                "missing_knowledge": 1,
                "hallucination": 1,
                "tool_failure": 1,
                "handoff": 1,
            },
        )
        self.assertTrue(all(url == "http://agent.local/ask" for url, _payload in calls))

    def test_generate_traffic_rejects_invalid_count(self) -> None:
        with self.assertRaisesRegex(TrafficError, "count"):
            generate_traffic.generate_traffic(count=0)

    def test_main_returns_nonzero_with_clear_error(self) -> None:
        with patch.object(generate_traffic, "generate_traffic", side_effect=TrafficError("boom")):
            with patch("sys.stderr") as stderr:
                status = generate_traffic.main(["--count", "1"])

        self.assertEqual(status, 1)
        self.assertIn("Traffic generation failed", "".join(call.args[0] for call in stderr.write.call_args_list))

    def test_main_prints_summary(self) -> None:
        with patch.object(
            generate_traffic,
            "generate_traffic",
            return_value=generate_traffic.TrafficSummary(total_sent=2, by_kind={"success": 2}),
        ):
            with patch("builtins.print") as fake_print:
                status = generate_traffic.main(["--count", "2"])

        self.assertEqual(status, 0)
        fake_print.assert_called_once_with("Demo traffic completed: sent=2, success=2")


if __name__ == "__main__":
    unittest.main()
