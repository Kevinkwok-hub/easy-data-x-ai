from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DashboardConfigTest(unittest.TestCase):
    def test_docker_compose_defines_required_services_and_ports(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]

        self.assertEqual(set(services), {"agent", "prometheus", "grafana"})
        self.assertEqual(services["agent"]["ports"], ["8000:8000"])
        self.assertEqual(services["prometheus"]["ports"], ["9090:9090"])
        self.assertEqual(services["grafana"]["ports"], ["3000:3000"])
        self.assertEqual(services["agent"]["environment"]["ENVIRONMENT"], "docker")

    def test_prometheus_scrapes_agent_metrics_endpoint(self) -> None:
        config = yaml.safe_load((ROOT / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
        jobs = {item["job_name"]: item for item in config["scrape_configs"]}
        agent_job = jobs["p5-knowledge-agent"]

        self.assertEqual(agent_job["metrics_path"], "/metrics")
        self.assertEqual(agent_job["static_configs"][0]["targets"], ["agent:8000"])

    def test_grafana_provisioning_loads_datasource_and_dashboard(self) -> None:
        datasource = yaml.safe_load(
            (ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text(encoding="utf-8")
        )
        dashboard = yaml.safe_load(
            (ROOT / "grafana" / "provisioning" / "dashboards" / "dashboard.yml").read_text(encoding="utf-8")
        )

        self.assertEqual(datasource["datasources"][0]["name"], "Prometheus")
        self.assertEqual(datasource["datasources"][0]["uid"], "Prometheus")
        self.assertEqual(datasource["datasources"][0]["url"], "http://prometheus:9090")
        self.assertEqual(dashboard["providers"][0]["options"]["path"], "/var/lib/grafana/dashboards")

    def test_dashboard_contains_required_panels_and_prometheus_queries(self) -> None:
        dashboard = json.loads((ROOT / "grafana" / "dashboards" / "agent_metrics.json").read_text(encoding="utf-8"))
        panels = {panel["title"]: panel for panel in dashboard["panels"]}

        required_titles = {
            "总览 / Overview",
            "数据层 / Data Layer",
            "模型层 / Model Layer",
            "业务层 / Business Layer",
            "运行层补充 / Runtime Supplement",
            "任务成功率 / Task Success Rate",
            "转人工率 / Human Handoff Rate",
            "检索故障率 / Retrieval Failure Rate",
            "回答评测覆盖率 / Answer Evaluation Coverage",
            "工具成功率 / Tool Success Rate",
        }
        self.assertEqual(dashboard["title"], "Agent 三层质量与运行监控 / Three-Layer Quality & Runtime")
        self.assertTrue(required_titles.issubset(panels))

        metric_panels = [panel for panel in dashboard["panels"] if panel["type"] != "row"]
        expressions = "\n".join(target["expr"] for panel in metric_panels for target in panel["targets"])
        for metric_name in [
            "agent_requests_total",
            "agent_tasks_total",
            "agent_task_success_total",
            "agent_handoff_total",
            "agent_request_duration_seconds_bucket",
            "agent_errors_total",
            "agent_retrieval_hit_total",
            "agent_retrieval_errors_total",
            "agent_knowledge_evaluated_total",
            "agent_answer_correct_total",
            "agent_answer_evaluated_total",
            "agent_hallucination_total",
            "agent_tool_calls_total",
            "agent_tool_errors_total",
            "agent_cost_total",
        ]:
            self.assertIn(metric_name, expressions)

        rate_titles = {
            "任务成功率 / Task Success Rate", "转人工率 / Human Handoff Rate",
            "知识覆盖率 / Knowledge Coverage Rate", "检索命中率 / Retrieval Hit Rate",
            "检索故障率 / Retrieval Failure Rate", "回答准确率 / Answer Accuracy",
            "回答评测覆盖率 / Answer Evaluation Coverage", "幻觉率 / Hallucination Rate",
            "工具成功率 / Tool Success Rate", "错误率 / Error Rate",
        }
        for title in rate_titles:
            self.assertIn("increase(", panels[title]["targets"][0]["expr"])
            self.assertEqual(panels[title]["fieldConfig"]["defaults"]["noValue"], "暂无可评测数据")


if __name__ == "__main__":
    unittest.main()
