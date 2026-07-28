import ast
import contextlib
import io
import json
import runpy
import time
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from d4_5_multi_user_isolation import IsolatedMemoryStore
from d4_6_memory_compression import FactMemoryStore
from d4_7_hot_cold_tier import TieredMemoryStore, migrate_if_needed
from d4_8_memory_cleanup import LifecycleMemoryStore, cleanup_memories


D4_DIR = Path(__file__).resolve().parent


def load_functions(file_name: str) -> dict:
    """只加载示例中的函数定义，避免测试触发真实模型或数据库初始化。"""
    source_path = D4_DIR / file_name
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module = ast.Module(body=function_nodes, type_ignores=[])
    namespace = {
        "json": json,
        "time": time,
        "uuid": uuid,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


class FakeCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def add(self, *, ids, documents, metadatas) -> None:
        self.ids.extend(ids)

    def count(self) -> int:
        return len(self.ids)


class FakeCompletions:
    def __init__(self, responses) -> None:
        self.responses = list(responses)

    def create(self, **kwargs):
        return self.responses.pop(0)


def fake_client(*responses):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(responses),
        )
    )


def response_with(message=None, *, choices=True):
    if not choices:
        return SimpleNamespace(choices=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def plain_message(content: str | None):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_message(name: str, arguments: str):
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(content=None, tool_calls=[call])


def tool_batch_message(*calls):
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
            for index, (name, arguments) in enumerate(calls, start=1)
        ],
    )


class MemoryIdSecurityTests(unittest.TestCase):
    def test_all_d4_memory_stores_generate_uuid4_ids(self):
        generated_ids: list[str] = []

        for file_name, function_name in (
            ("d4_3_with_memory.py", "add_memory"),
            ("d4_4_memory_agent.py", "save_memory"),
        ):
            with self.subTest(file_name=file_name):
                namespace = load_functions(file_name)
                collection = FakeCollection()
                namespace["memory_col"] = collection
                namespace[function_name](["用户偏好简洁回答", "用户是 Python 开发者"])
                generated_ids.extend(collection.ids)

        isolated = IsolatedMemoryStore()
        generated_ids.append(isolated.add("事实", user_id="alice").id)

        tiered = TieredMemoryStore()
        generated_ids.append(
            tiered.add(
                "事实",
                user_id="alice",
                retention=0.9,
                days_since_access=1,
            ).id
        )

        lifecycle = LifecycleMemoryStore()
        generated_ids.append(
            lifecycle.add(
                "事实",
                user_id="alice",
                created_at=datetime(2026, 7, 28, 12, 0, 0),
            ).id
        )

        self.assertEqual(len(generated_ids), len(set(generated_ids)))
        for memory_id in generated_ids:
            parsed = uuid.UUID(memory_id)
            self.assertEqual(parsed.version, 4)


class ExternalInitializationSafetyTests(unittest.TestCase):
    def test_importing_database_examples_has_no_external_side_effects(self):
        for file_name in ("d4_3_with_memory.py", "d4_4_memory_agent.py"):
            with self.subTest(file_name=file_name):
                with (
                    patch("openai.OpenAI", side_effect=AssertionError("导入时不应创建模型客户端")),
                    patch("pyseekdb.Client", side_effect=AssertionError("导入时不应打开数据库")),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    namespace = runpy.run_path(
                        str(D4_DIR / file_name),
                        run_name=f"d4_import_test_{file_name}",
                    )

                self.assertIsNone(namespace["client"])
                self.assertIsNone(namespace["db"])
                self.assertIsNone(namespace["memory_col"])

    def test_memory_database_paths_are_anchored_to_script_directory(self):
        expected_paths = {
            "d4_3_with_memory.py": D4_DIR / "memory.db",
            "d4_4_memory_agent.py": D4_DIR / "memory_persistent.db",
        }

        for file_name, expected_path in expected_paths.items():
            with self.subTest(file_name=file_name):
                with (
                    patch("openai.OpenAI", side_effect=AssertionError("导入时不应创建模型客户端")),
                    patch("pyseekdb.Client", side_effect=AssertionError("导入时不应打开数据库")),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    namespace = runpy.run_path(
                        str(D4_DIR / file_name),
                        run_name=f"d4_path_test_{file_name}",
                    )
                self.assertEqual(namespace["MEMORY_DB_PATH"], expected_path)

    def test_explicit_initialization_accepts_injected_client_and_database(self):
        class FakeDatabase:
            def __init__(self, *, exists: bool) -> None:
                self.exists = exists
                self.collection = FakeCollection()
                self.calls: list[str] = []

            def has_collection(self, name: str) -> bool:
                self.calls.append(f"has:{name}")
                return self.exists

            def delete_collection(self, name: str) -> None:
                self.calls.append(f"delete:{name}")

            def create_collection(self, name: str):
                self.calls.append(f"create:{name}")
                return self.collection

            def get_collection(self, name: str):
                self.calls.append(f"get:{name}")
                return self.collection

        cases = (
            (
                "d4_3_with_memory.py",
                True,
                [
                    "has:user_memory_demo",
                    "delete:user_memory_demo",
                    "create:user_memory_demo",
                ],
            ),
            (
                "d4_4_memory_agent.py",
                True,
                [
                    "has:user_memory_persistent",
                    "get:user_memory_persistent",
                ],
            ),
        )

        for file_name, exists, expected_calls in cases:
            with self.subTest(file_name=file_name):
                with (
                    patch("openai.OpenAI", side_effect=AssertionError("导入时不应创建模型客户端")),
                    patch("pyseekdb.Client", side_effect=AssertionError("导入时不应打开数据库")),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    namespace = runpy.run_path(
                        str(D4_DIR / file_name),
                        run_name=f"d4_init_test_{file_name}",
                    )
                    model_client = object()
                    database = FakeDatabase(exists=exists)
                    collection = namespace["initialize_runtime"](
                        api_client=model_client,
                        database=database,
                    )

                runtime_globals = namespace["initialize_runtime"].__globals__
                self.assertIs(runtime_globals["client"], model_client)
                self.assertIs(runtime_globals["db"], database)
                self.assertIs(runtime_globals["memory_col"], collection)
                self.assertEqual(database.calls, expected_calls)


class IsolatedReadSecurityTests(unittest.TestCase):
    def test_add_returns_copy_that_cannot_change_owner(self):
        store = IsolatedMemoryStore()
        created = store.add("Alice 的私密事实", user_id="alice")
        created.user_id = "bob"
        created.shared_with["bob"] = ["write"]

        with self.assertRaises(PermissionError):
            store.delete(created.id, requester_id="bob")

        fresh = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh.user_id, "alice")
        self.assertEqual(fresh.shared_with, {})

    def test_get_requires_authorized_requester_and_returns_deep_copy(self):
        store = IsolatedMemoryStore()
        created = store.add("Alice 的私密事实", user_id="alice")

        with self.assertRaises(TypeError):
            store.get(created.id)
        with self.assertRaises(PermissionError):
            store.get(created.id, requester_id="bob")

        owner_copy = store.get(created.id, requester_id="alice")
        self.assertIsNotNone(owner_copy)
        owner_copy.content = "被外部篡改"
        owner_copy.shared_with["bob"] = ["write"]

        fresh_copy = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh_copy.content, "Alice 的私密事实")
        self.assertEqual(fresh_copy.shared_with, {})

    def test_search_and_user_listing_return_defensive_copies(self):
        store = IsolatedMemoryStore()
        created = store.add("Alice 使用 Python", user_id="alice")

        search_copy = store.search("Python", user_id="alice")[0]
        search_copy.content = "搜索结果被篡改"

        listing_copy = store.get_all_for_user(
            "alice",
            requester_id="alice",
        )[0]
        listing_copy.content = "列表结果被篡改"

        fresh_copy = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh_copy.content, "Alice 使用 Python")

    def test_user_listing_requires_matching_requester(self):
        store = IsolatedMemoryStore()
        store.add("Alice 的私密事实", user_id="alice")

        with self.assertRaises(TypeError):
            store.get_all_for_user("alice")

        try:
            store.get_all_for_user("alice", requester_id="bob")
        except Exception as exc:
            self.assertIsInstance(exc, PermissionError)
        else:
            self.fail("cross-user listing must be denied")

    def test_insecure_search_demo_returns_deep_copies(self):
        store = IsolatedMemoryStore()
        created = store.add("Alice 对花生过敏", user_id="alice")

        leaked_copy = store.search_without_isolation("花生 过敏")[0]
        leaked_copy.content = "反例读取结果被篡改"
        leaked_copy.shared_with["mallory"] = ["write"]

        fresh_copy = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh_copy.content, "Alice 对花生过敏")
        self.assertEqual(fresh_copy.shared_with, {})


class TieredAndLifecycleReadSecurityTests(unittest.TestCase):
    def test_tiered_migration_atomically_syncs_copy_metrics(self):
        store = TieredMemoryStore()
        memory = store.add(
            "Alice 的温层事实",
            user_id="alice",
            retention=0.5,
            days_since_access=20,
        )
        self.assertEqual(memory.tier, "warm")

        memory.retention = 0.2
        memory.days_since_access = 45
        moved = migrate_if_needed(store, memory)

        self.assertEqual(moved, "warm -> cold")
        archived = store.archive[memory.id]
        self.assertEqual(archived.tier, "cold")
        self.assertEqual(archived.retention, 0.2)
        self.assertEqual(archived.days_since_access, 45)
        self.assertFalse(archived.in_hot_index)

    def test_tiered_add_returns_copy_that_cannot_change_owner(self):
        store = TieredMemoryStore()
        created = store.add(
            "Alice 使用 Python",
            user_id="alice",
            retention=0.9,
            days_since_access=1,
        )
        created.user_id = "bob"
        created.retention = 0.0

        with self.assertRaises(PermissionError):
            store.get(created.id, requester_id="bob")

        fresh = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh.user_id, "alice")
        self.assertEqual(fresh.retention, 0.9)

    def test_lifecycle_add_copy_cannot_disable_protection(self):
        now = datetime(2026, 7, 28, 12, 0, 0)
        store = LifecycleMemoryStore()
        created = store.add(
            "primary_language=Python",
            user_id="alice",
            created_at=now,
            protected=True,
        )
        created.protected = False

        with self.assertRaises(PermissionError):
            store.delete(created.id, requester_id="alice")

        self.assertTrue(store.records[created.id].protected)

    def test_tiered_get_requires_owner_and_read_results_are_copies(self):
        store = TieredMemoryStore()
        created = store.add(
            "Alice 使用 Python",
            user_id="alice",
            retention=0.9,
            days_since_access=1,
        )

        with self.assertRaises(PermissionError):
            store.get(created.id, requester_id="bob")

        get_copy = store.get(created.id, requester_id="alice")
        get_copy.content = "按 ID 读取结果被篡改"

        search_copy = store.search("Python", user_id="alice")[0]
        search_copy.content = "搜索结果被篡改"

        fresh_copy = store.get(created.id, requester_id="alice")
        self.assertEqual(fresh_copy.content, "Alice 使用 Python")

    def test_lifecycle_iteration_requires_user_and_returns_copies(self):
        store = LifecycleMemoryStore()
        created = store.add(
            "Alice 的生命周期事实",
            user_id="alice",
            created_at=datetime(2026, 7, 28, 12, 0, 0),
        )

        with self.assertRaises(TypeError):
            list(store.iter_all())

        item_copy = list(store.iter_all(user_id="alice"))[0]
        item_copy.content = "遍历结果被篡改"

        self.assertEqual(store.records[created.id].content, "Alice 的生命周期事实")


class DistillationSecurityTests(unittest.TestCase):
    def test_consolidation_requires_explicit_non_empty_fact(self):
        store = FactMemoryStore(
            facts=[
                "用户认为 Flask 偏慢",
                "用户准备迁移到 FastAPI",
                "用户后端已切到 FastAPI",
            ]
        )

        with self.assertRaises(TypeError):
            store.consolidate_topic(["Flask", "FastAPI"], min_count=3)
        with self.assertRaises(ValueError):
            store.consolidate_topic(
                ["Flask", "FastAPI"],
                distilled_fact="   ",
                min_count=3,
            )

    def test_consolidation_uses_the_caller_provided_fact(self):
        store = FactMemoryStore(
            facts=[
                "用户认为 Flask 偏慢",
                "用户准备迁移到 FastAPI",
                "用户后端已切到 FastAPI",
            ]
        )

        distilled = store.consolidate_topic(
            ["Flask", "FastAPI"],
            distilled_fact="用户已完成 Web 框架迁移",
            min_count=3,
        )

        self.assertEqual(distilled, "用户已完成 Web 框架迁移")
        self.assertEqual(store.facts, ["用户已完成 Web 框架迁移"])


class CleanupSecurityTests(unittest.TestCase):
    def test_cleanup_requires_non_empty_user_id(self):
        now = datetime(2026, 7, 28, 12, 0, 0)
        store = LifecycleMemoryStore()
        store.add(
            "过期事实",
            user_id="alice",
            created_at=now - timedelta(days=200),
            last_accessed_at=now - timedelta(days=150),
        )

        with self.assertRaises(TypeError):
            cleanup_memories(store, now)
        with self.assertRaises(ValueError):
            cleanup_memories(store, now, user_id="   ")

    def test_direct_delete_checks_owner_and_protected_flag(self):
        now = datetime(2026, 7, 28, 12, 0, 0)
        store = LifecycleMemoryStore()
        protected = store.add(
            "primary_language=Python",
            user_id="alice",
            created_at=now - timedelta(days=200),
            protected=True,
        )

        with self.assertRaises(PermissionError):
            store.delete(protected.id, requester_id="bob")
        with self.assertRaises(PermissionError):
            store.delete(protected.id, requester_id="alice")

        self.assertIn(protected.id, store.records)
        self.assertIn(protected.id, store.vector_index)

    def test_cleanup_only_changes_the_requested_user(self):
        now = datetime(2026, 7, 28, 12, 0, 0)
        store = LifecycleMemoryStore()
        alice = store.add(
            "Alice 的过期事实",
            user_id="alice",
            created_at=now - timedelta(days=200),
            last_accessed_at=now - timedelta(days=150),
        )
        bob = store.add(
            "Bob 的过期事实",
            user_id="bob",
            created_at=now - timedelta(days=200),
            last_accessed_at=now - timedelta(days=150),
        )

        cleanup_memories(store, now, user_id="alice", dry_run=False)

        self.assertNotIn(alice.id, store.records)
        self.assertIn(bob.id, store.records)


class ToolCallSecurityTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_functions("d4_1_react_loop.py")
        self.namespace["MODEL"] = "offline-test-model"
        self.namespace["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "safe_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ]

    def test_unknown_tool_name_is_rejected_before_execution(self):
        executed: list[str] = []
        self.namespace["tool_functions"] = {
            "safe_tool": lambda args: executed.append("safe") or "ok",
        }
        self.namespace["client"] = fake_client(
            response_with(tool_message("delete_everything", "{}")),
        )

        with self.assertRaisesRegex(ValueError, "未注册"):
            self.namespace["agent_loop"]("问题")

        self.assertEqual(executed, [])

    def test_entire_tool_batch_is_validated_before_any_execution(self):
        executed: list[str] = []
        self.namespace["tool_functions"] = {
            "safe_tool": lambda args: executed.append("safe") or "ok",
        }
        self.namespace["client"] = fake_client(
            response_with(
                tool_batch_message(
                    ("safe_tool", '{"value": "ok"}'),
                    ("delete_everything", "{}"),
                )
            ),
        )

        with self.assertRaisesRegex(ValueError, "未注册"):
            self.namespace["agent_loop"]("问题")

        self.assertEqual(executed, [])

    def test_malformed_or_non_object_json_arguments_are_rejected(self):
        for arguments in ('{"city":', '["北京"]'):
            with self.subTest(arguments=arguments):
                executed: list[str] = []
                self.namespace["tool_functions"] = {
                    "get_weather": lambda args: executed.append("called") or "ok",
                }
                self.namespace["client"] = fake_client(
                    response_with(tool_message("get_weather", arguments)),
                    response_with(plain_message("不会执行到这里")),
                )

                with self.assertRaisesRegex(ValueError, "JSON|对象"):
                    self.namespace["agent_loop"]("问题")

                self.assertEqual(executed, [])

    def test_missing_or_non_string_required_arguments_are_rejected_before_execution(self):
        for arguments in ("{}", '{"city": 42}'):
            with self.subTest(arguments=arguments):
                executed: list[str] = []
                self.namespace["tool_functions"] = {
                    "get_weather": lambda args: executed.append("called") or "ok",
                }
                self.namespace["client"] = fake_client(
                    response_with(tool_message("get_weather", arguments)),
                )

                with self.assertRaisesRegex(ValueError, "必填|字符串"):
                    self.namespace["agent_loop"]("问题")

                self.assertEqual(executed, [])

    def test_invalid_later_arguments_prevent_entire_batch_execution(self):
        executed: list[str] = []
        self.namespace["tool_functions"] = {
            "safe_tool": lambda args: executed.append("safe") or "ok",
            "get_weather": lambda args: executed.append("weather") or "ok",
        }
        self.namespace["client"] = fake_client(
            response_with(
                tool_batch_message(
                    ("safe_tool", '{"value": "ok"}'),
                    ("get_weather", "{}"),
                )
            ),
            response_with(plain_message("不会执行到这里")),
        )

        with self.assertRaisesRegex(ValueError, "必填"):
            self.namespace["agent_loop"]("问题")

        self.assertEqual(executed, [])

    def test_empty_or_contentless_model_response_is_rejected(self):
        self.namespace["tool_functions"] = {}

        for response in (
            response_with(choices=False),
            response_with(None),
            response_with(plain_message(None)),
        ):
            with self.subTest(response=response):
                self.namespace["client"] = fake_client(response)
                with self.assertRaisesRegex(RuntimeError, "空响应"):
                    self.namespace["agent_loop"]("问题")

    def test_valid_tool_call_still_returns_final_answer(self):
        self.namespace["tool_functions"] = {
            "get_weather": lambda args: f"{args['city']}：晴",
        }
        self.namespace["client"] = fake_client(
            response_with(tool_message("get_weather", '{"city": "北京"}')),
            response_with(plain_message("北京今天晴。")),
        )

        answer = self.namespace["agent_loop"]("北京天气？")

        self.assertEqual(answer, "北京今天晴。")


if __name__ == "__main__":
    unittest.main()
