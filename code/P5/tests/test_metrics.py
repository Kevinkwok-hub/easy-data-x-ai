from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from app.agent import KnowledgeAgent
from app.main import AskRequest, app
from app.observability.prometheus_metrics import AgentMetrics, validate_label_names


class PrometheusMetricsTest(unittest.TestCase):
    def test_metrics_counter_growth_for_agent_result(self) -> None:
        registry = CollectorRegistry()
        metrics = AgentMetrics(registry=registry, agent_version="test", environment="unit")
        result = KnowledgeAgent().run("请导出用量 CSV。")

        metrics.record_agent_result(result, expected_answer_contains="工具执行失败")
        rendered = metrics.render_latest().decode("utf-8")

        self.assertIn('agent_tasks_total{agent_version="test",environment="unit"} 1.0', rendered)
        self.assertIn('agent_handoff_total{agent_version="test",environment="unit"} 1.0', rendered)
        self.assertIn('agent_tool_calls_total{agent_version="test",environment="unit",tool_name="mock_tool"} 1.0', rendered)
        self.assertIn('agent_tool_errors_total{agent_version="test",environment="unit",tool_name="mock_tool"} 1.0', rendered)
        self.assertIn('agent_answer_correct_total{agent_version="test",environment="unit"} 1.0', rendered)
        self.assertIn('agent_answer_evaluated_total{agent_version="test",environment="unit"} 1.0', rendered)

    def test_answer_accuracy_only_counts_annotated_requests(self) -> None:
        metrics = AgentMetrics(registry=CollectorRegistry(), agent_version="test", environment="unit")
        result = KnowledgeAgent().run("退款要在多久内提交？")
        metrics.record_agent_result(result)
        metrics.record_agent_result(result, expected_answer_contains="7 天内")
        metrics.record_agent_result(result, expected_answer_contains="错误事实")
        rendered = metrics.render_latest().decode("utf-8")
        self.assertIn('agent_answer_evaluated_total{agent_version="test",environment="unit"} 2.0', rendered)
        self.assertIn('agent_answer_correct_total{agent_version="test",environment="unit"} 1.0', rendered)

    def test_retrieval_denominators_exclude_failures_and_missing_knowledge(self) -> None:
        metrics = AgentMetrics(registry=CollectorRegistry(), agent_version="test", environment="unit")
        agent = KnowledgeAgent()
        for query in (
            "退款要在多久内提交？",
            "simulate_retrieval_miss：企业版 SLA 是多少？",
            "simulate_retrieval_failure：帮我查询 SLA。",
            "海外仓冷链温控方案怎么配置？",
        ):
            metrics.record_agent_result(agent.run(query))
        rendered = metrics.render_latest().decode("utf-8")
        self.assertIn('agent_knowledge_evaluated_total{agent_version="test",environment="unit"} 3.0', rendered)
        self.assertIn('agent_knowledge_available_total{agent_version="test",environment="unit"} 2.0', rendered)
        self.assertIn('agent_retrieval_total{agent_version="test",environment="unit"} 2.0', rendered)
        self.assertIn('agent_retrieval_hit_total{agent_version="test",environment="unit"} 1.0', rendered)
        self.assertIn('agent_retrieval_errors_total{agent_version="test",environment="unit"} 1.0', rendered)

    def test_sensitive_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            validate_label_names(("agent_version", "environment", "query"))

    def test_metrics_default_labels_can_come_from_environment(self) -> None:
        with patch.dict("os.environ", {"AGENT_VERSION": "env-agent", "ENVIRONMENT": "staging"}):
            metrics = AgentMetrics(registry=CollectorRegistry())
        result = KnowledgeAgent().run("退款要在多久内提交？")

        metrics.record_agent_result(result)
        rendered = metrics.render_latest().decode("utf-8")

        self.assertIn('agent_tasks_total{agent_version="env-agent",environment="staging"} 1.0', rendered)

    def test_ask_endpoint_returns_agent_result_and_updates_metrics(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/ask",
            json={
                "query": "退款要在多久内提交？",
                "task_id": "api-test-001",
                "expected_answer_contains": "7 天内",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["task_id"], "api-test-001")
        self.assertTrue(data["retrieval_hit"])
        self.assertFalse(data["retrieval_failed"])
        self.assertTrue(data["task_success"])

        metrics_response = client.get("/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text
        self.assertIn("agent_requests_total", metrics_text)
        self.assertIn("agent_tasks_total", metrics_text)
        self.assertIn("agent_retrieval_hit_total", metrics_text)

    def test_health_endpoint(self) -> None:
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ask_endpoint_rejects_oversized_fields(self) -> None:
        client = TestClient(app)
        limits = {
            "query": 4096,
            "task_id": 128,
            "expected_answer_contains": 1024,
        }

        for field, limit in limits.items():
            with self.subTest(field=field):
                payload = {"query": "有效问题"}
                payload[field] = "x" * (limit + 1)
                response = client.post("/ask", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_ask_endpoint_rejects_blank_query_and_expected_answer(self) -> None:
        client = TestClient(app)
        for payload in (
            {"query": "   "},
            {"query": "有效问题", "expected_answer_contains": " \t "},
        ):
            with self.subTest(payload=payload):
                response = client.post("/ask", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_ask_request_normalizes_surrounding_query_whitespace(self) -> None:
        request = AskRequest(query="  退款要在多久内提交？  ")
        self.assertEqual(request.query, "退款要在多久内提交？")


if __name__ == "__main__":
    unittest.main()
