from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent import KnowledgeAgent
from app.main import app
from app.observability.langsmith_tracing import LangSmithConfig, LangSmithTracer


class FakeRun:
    """测试用 LangSmith run 替身，用来保存 metadata 和 end(outputs)。"""

    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        """复制 metadata，避免测试过程中调用方和 fake run 共享同一个可变字典。"""
        self.metadata = dict(metadata or {})
        self.outputs: dict[str, object] | None = None

    def end(self, *, outputs: dict[str, object] | None = None) -> None:
        """模拟 LangSmith SDK 的 run.end，记录最终 outputs 供断言使用。"""
        self.outputs = outputs or {}


class FakeTraceContext:
    """测试用 trace context manager，进入时把 run 记录到 FakeTraceFactory。"""

    def __init__(self, recorder: "FakeTraceFactory", kwargs: dict[str, object]) -> None:
        """保存 trace 调用参数，便于测试检查 run name、project 和 parent。"""
        self.recorder = recorder
        self.kwargs = kwargs
        self.run = FakeRun(kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None)

    def __enter__(self) -> FakeRun:
        """模拟进入 LangSmith trace，上下文开始时记录一次调用。"""
        self.recorder.calls.append({"kwargs": self.kwargs, "run": self.run})
        return self.run

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """测试 fake 不吞异常，让失败按正常 unittest 方式暴露。"""
        return None


class FakeTraceFactory:
    """测试用 trace factory，替代真实 LangSmith SDK，避免网络和 API key 依赖。"""

    def __init__(self) -> None:
        """保存所有 trace 调用，供测试断言 Trace 树结构。"""
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> FakeTraceContext:
        """兼容真实 `trace(\"Run Name\", ...)` 的调用方式。"""
        if args:
            kwargs = {**kwargs, "name": args[0]}
        return FakeTraceContext(self, kwargs)


class LangSmithTracingTest(unittest.TestCase):
    def test_disabled_tracing_runs_agent_without_trace_factory(self) -> None:
        def fail_if_called(*args: object, **kwargs: object) -> FakeTraceContext:
            raise AssertionError("trace factory should not be called when tracing is disabled")

        tracer = LangSmithTracer(
            LangSmithConfig(tracing=False, api_key=""),
            trace_factory=fail_if_called,
        )
        result = tracer.trace_agent_run(
            query="退款要在多久内提交？",
            task_id="trace-off",
            expected_answer_contains="7 天内",
            run_agent=lambda: KnowledgeAgent().run("退款要在多久内提交？"),
        )

        self.assertTrue(result.task_success)
        self.assertEqual(result.matched_doc_id, "kb-general-refund")

    def test_true_tracing_without_api_key_is_noop(self) -> None:
        with patch.dict(
            "os.environ",
            {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""},
            clear=False,
        ):
            config = LangSmithConfig.from_env()

        self.assertFalse(config.enabled)
        self.assertTrue(config.tracing)
        self.assertEqual(config.api_key, "")

    def test_enabled_tracing_creates_expected_runs_and_metadata(self) -> None:
        trace_factory = FakeTraceFactory()
        tracer = LangSmithTracer(
            LangSmithConfig(
                tracing=True,
                api_key="test-key",
                project="unit-project",
                agent_version="test-agent",
                environment="unit",
            ),
            trace_factory=trace_factory,
        )

        result = tracer.trace_agent_run(
            query="请导出用量 CSV。",
            task_id="trace-tool",
            expected_answer_contains="工具执行失败",
            run_agent=lambda: KnowledgeAgent().run("请导出用量 CSV。"),
        )

        self.assertTrue(result.tool_called)
        self.assertEqual(
            [call["kwargs"]["name"] for call in trace_factory.calls],
            ["Agent Trace", "Retrieval Run", "Generation Run", "Tool Run", "Final Response"],
        )
        root_run = trace_factory.calls[0]["run"]
        self.assertIsInstance(root_run, FakeRun)
        self.assertEqual(root_run.metadata["agent_version"], "test-agent")
        self.assertEqual(root_run.metadata["environment"], "unit")
        self.assertEqual(root_run.metadata["task_id"], "trace-tool")
        self.assertEqual(root_run.metadata["category"], "tool")
        self.assertEqual(root_run.metadata["evaluator_score"], 1.0)

        for call in trace_factory.calls:
            kwargs = call["kwargs"]
            self.assertEqual(kwargs["project_name"], "unit-project")

    def test_metrics_labels_do_not_include_trace_or_payload_fields(self) -> None:
        client = TestClient(app)
        client.post(
            "/ask",
            json={
                "query": "请导出用量 CSV。",
                "task_id": "sensitive-task-id",
                "expected_answer_contains": "工具执行失败",
            },
        )

        metrics_text = client.get("/metrics").text
        self.assertNotIn("query=", metrics_text)
        self.assertNotIn("answer=", metrics_text)
        self.assertNotIn("trace_id=", metrics_text)
        self.assertNotIn("task_id=", metrics_text)
        self.assertNotIn("sensitive-task-id", metrics_text)


if __name__ == "__main__":
    unittest.main()
