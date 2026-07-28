import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk

from config import Config


D1_MODULES = (
    "D1.d1_1_base",
    "D1.d1_2_multi_turn",
    "D1.d1_3_streaming",
    "D1.d1_4_tool_use_mock",
    "D1.d1_5_tool_use_seekdb",
    "D1.d1_6_agent",
)


class FakeChatModel:
    def __init__(self, content="测试回答"):
        self.content = content
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.content)

    def stream(self, messages):
        self.invocations.append(messages)
        return iter([AIMessageChunk(content=self.content)])

    def bind_tools(self, _tools):
        return self


class FakeCollection:
    def add(self, **_kwargs):
        return None

    def query(self, **_kwargs):
        return {
            "documents": [["测试知识"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }


class FakeDatabase:
    def __init__(self, cleanup_error=None):
        self.collection = FakeCollection()
        self.cleanup_error = cleanup_error
        self.events = []
        self.exists = False

    def has_collection(self, name):
        self.events.append(("has", name))
        return self.exists

    def create_collection(self, name):
        self.events.append(("create", name))
        self.exists = True
        return self.collection

    def delete_collection(self, name):
        self.events.append(("delete", name))
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.exists = False

    def close(self):
        self.events.append(("close",))


class ContextOnlyDatabase(FakeDatabase):
    close = None

    def __exit__(self, *_args):
        self.events.append(("exit",))


class FakeAgent:
    def stream(self, _payload):
        return iter([{"model": {"messages": [AIMessage(content="Agent 测试回答")]}}])


class D1ImportSafetyTests(unittest.TestCase):
    def test_importing_all_examples_does_not_initialize_models_agents_or_databases(self):
        for module_name in D1_MODULES:
            with self.subTest(module=module_name):
                sys.modules.pop(module_name, None)
                with (
                    patch(
                        "langchain.chat_models.init_chat_model",
                        side_effect=AssertionError("导入时不应初始化模型"),
                    ),
                    patch(
                        "langchain.agents.create_agent",
                        side_effect=AssertionError("导入时不应创建 Agent"),
                    ),
                    patch(
                        "pyseekdb.Client",
                        side_effect=AssertionError("导入时不应打开数据库"),
                    ),
                ):
                    module = importlib.import_module(module_name)

                self.assertTrue(callable(module.main))
                self.assertTrue(callable(module.run_demo))


class D1ApiKeyBoundaryTests(unittest.TestCase):
    def test_basic_chat_entries_keep_the_course_model_configuration(self):
        """只验证课程模型名未被测试时的临时替换污染，不评价模型效果。"""
        for module_name in (
            "D1.d1_1_base",
            "D1.d1_2_multi_turn",
            "D1.d1_3_streaming",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                selected_models = []

                def model_factory(model_name, **_kwargs):
                    selected_models.append(model_name)
                    return FakeChatModel()

                with (
                    patch.object(Config, "SILICONFLOW_API_KEY", "sk-test"),
                    redirect_stdout(io.StringIO()),
                ):
                    module.main(model_factory=model_factory)

                self.assertEqual(
                    selected_models,
                    ["tencent/Hunyuan-MT-7B"],
                )

    def test_missing_key_stops_before_model_and_database_initialization(self):
        for module_name, key_name in (
            ("D1.d1_1_base", "SILICONFLOW_API_KEY"),
            ("D1.d1_2_multi_turn", "SILICONFLOW_API_KEY"),
            ("D1.d1_3_streaming", "SILICONFLOW_API_KEY"),
            ("D1.d1_4_tool_use_mock", "SILICONFLOW_API_KEY"),
            ("D1.d1_5_tool_use_seekdb", "SILICONFLOW_API_KEY"),
            ("D1.d1_6_agent", "DASHSCOPE_API_KEY"),
        ):
            module = importlib.import_module(module_name)
            output = io.StringIO()
            kwargs = {
                "model_factory": lambda *_args, **_kwargs: self.fail("不应初始化模型"),
            }
            if module_name.endswith(("d1_5_tool_use_seekdb", "d1_6_agent")):
                kwargs["client_factory"] = (
                    lambda **_kwargs: self.fail("缺少 Key 时不应打开数据库")
                )

            with (
                patch.object(Config, key_name, "YOUR_API_KEY"),
                redirect_stdout(output),
            ):
                status = module.main(**kwargs)

            self.assertEqual(status, 1)
            self.assertIn("未配置", output.getvalue())


class D1SeekdbLifecycleTests(unittest.TestCase):
    def test_seekdb_entries_release_context_only_clients(self):
        for module_name in (
            "D1.d1_5_tool_use_seekdb",
            "D1.d1_6_agent",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                database = ContextOnlyDatabase()

                module._close_database(database)

                self.assertEqual(database.events, [("exit",)])

    def test_seekdb_entries_accept_cli_database_path_override(self):
        expected = Path("/tmp/d1-test-seekdb")

        for module_name in (
            "D1.d1_5_tool_use_seekdb",
            "D1.d1_6_agent",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                args = module.parse_args(["--db-path", str(expected)])
                self.assertEqual(args.db_path, expected)

    def test_d1_5_uses_overridden_path_and_always_cleans_up(self):
        module = importlib.import_module("D1.d1_5_tool_use_seekdb")
        database = FakeDatabase()
        received_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "seekdb"

            def client_factory(**kwargs):
                received_paths.append(kwargs["path"])
                return database

            with (
                patch.object(Config, "SILICONFLOW_API_KEY", "sk-test"),
                redirect_stdout(io.StringIO()),
            ):
                status = module.main(
                    model_factory=lambda *_args, **_kwargs: FakeChatModel(),
                    client_factory=client_factory,
                    db_path=db_path,
                )

        self.assertEqual(status, 0)
        self.assertEqual(received_paths, [str(db_path.resolve())])
        self.assertIn(("delete", "d1_knowledge_base"), database.events)
        self.assertEqual(database.events[-1], ("close",))

    def test_d1_5_does_not_swallow_cleanup_errors(self):
        module = importlib.import_module("D1.d1_5_tool_use_seekdb")
        database = FakeDatabase(cleanup_error=RuntimeError("清理失败"))

        with (
            patch.object(Config, "SILICONFLOW_API_KEY", "sk-test"),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "清理失败"),
        ):
            module.main(
                model_factory=lambda *_args, **_kwargs: FakeChatModel(),
                client_factory=lambda **_kwargs: database,
            )

    def test_d1_6_probes_configured_model_before_agent_and_cleans_up(self):
        module = importlib.import_module("D1.d1_6_agent")
        database = FakeDatabase()
        model = FakeChatModel(content="ok")
        model_calls = []

        def model_factory(model_name, **_kwargs):
            model_calls.append(model_name)
            return model

        with (
            patch.object(Config, "DASHSCOPE_API_KEY", "sk-test"),
            redirect_stdout(io.StringIO()),
        ):
            status = module.main(
                model_factory=model_factory,
                agent_factory=lambda **_kwargs: FakeAgent(),
                client_factory=lambda **_kwargs: database,
                questions=["测试问题"],
            )

        self.assertEqual(status, 0)
        self.assertEqual(model_calls, ["qwen-plus"])
        self.assertTrue(model.invocations, "创建 Agent 前应先执行真实模型探针")
        self.assertIn(("delete", "d1_agent_kb"), database.events)
        self.assertEqual(database.events[-1], ("close",))


if __name__ == "__main__":
    unittest.main()
