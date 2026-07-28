import contextlib
import io
import runpy
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


D4_DIR = Path(__file__).resolve().parent


def full_response(*, content="完成", choices=True, message=True):
    """构造完整的 OpenAI ChatCompletion 响应形状。"""
    if not choices:
        choice_items = []
    elif not message:
        choice_items = [SimpleNamespace(index=0, message=None, finish_reason="stop")]
    else:
        choice_items = [
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ]
    return SimpleNamespace(
        id="chatcmpl-offline",
        object="chat.completion",
        created=1_722_144_000,
        model="offline-model",
        choices=choice_items,
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
            total_tokens=16,
        ),
        system_fingerprint=None,
    )


def fake_client(*responses):
    response_queue = list(responses)

    def create(**_kwargs):
        if not response_queue:
            raise AssertionError("模型调用次数超出预期")
        return response_queue.pop(0)

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def load_module(file_name: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(
            str(D4_DIR / file_name),
            run_name=f"d4_runnability_{file_name}",
        )


class ImportAndInjectionTests(unittest.TestCase):
    def test_d4_1_and_d4_2_import_without_creating_client_or_running_demo(self):
        for file_name in ("d4_1_react_loop.py", "d4_2_no_memory.py"):
            with self.subTest(file_name=file_name):
                with (
                    patch(
                        "openai.OpenAI",
                        side_effect=AssertionError("导入时不应创建模型客户端"),
                    ),
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    namespace = runpy.run_path(
                        str(D4_DIR / file_name),
                        run_name=f"d4_import_{file_name}",
                    )

                self.assertEqual("", output.getvalue())
                self.assertIn("create_model_client", namespace)

    def test_d4_1_agent_loop_uses_injected_client(self):
        namespace = load_module("d4_1_react_loop.py")
        answer = namespace["agent_loop"](
            "你好",
            api_client=fake_client(full_response(content="你好！")),
        )
        self.assertEqual("你好！", answer)

    def test_d4_2_chat_uses_injected_client_and_complete_response(self):
        namespace = load_module("d4_2_no_memory.py")
        answer = namespace["chat_without_memory"](
            "你好",
            api_client=fake_client(full_response(content="你好！")),
        )
        self.assertEqual("你好！", answer)


class ModelResponseBoundaryTests(unittest.TestCase):
    def test_d4_1_rejects_missing_or_duplicate_tool_ids_before_execution(self):
        namespace = load_module("d4_1_react_loop.py")

        def response_with_calls(calls):
            response = full_response(content=None)
            response.choices[0].message.tool_calls = calls
            return response

        make_call = lambda call_id: SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"city": "北京"}',
            ),
        )
        for calls in (
            [make_call("")],
            [make_call("same"), make_call("same")],
        ):
            executed = []
            with (
                self.subTest(calls=calls),
                patch.dict(
                    namespace["tool_functions"],
                    {"get_weather": lambda args: executed.append(args) or "晴"},
                ),
                self.assertRaisesRegex(ValueError, "id"),
            ):
                namespace["agent_loop"](
                    "天气",
                    api_client=fake_client(response_with_calls(calls)),
                )
            self.assertEqual([], executed)

    def test_d4_2_rejects_missing_choices_message_or_content(self):
        namespace = load_module("d4_2_no_memory.py")
        for response in (
            full_response(choices=False),
            full_response(message=False),
            full_response(content=None),
            full_response(content="   "),
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(RuntimeError, "空响应"):
                    namespace["chat_without_memory"](
                        "你好",
                        api_client=fake_client(response),
                    )

    def test_fact_distillation_safely_degrades_on_missing_response_fields(self):
        cases = (
            ("d4_3_with_memory.py", "extract_facts_from_conversation"),
            ("d4_4_memory_agent.py", "extract_facts"),
        )
        for file_name, function_name in cases:
            namespace = load_module(file_name)
            for response in (
                full_response(choices=False),
                full_response(message=False),
                full_response(content=None),
                full_response(content="   "),
            ):
                with self.subTest(file_name=file_name, response=response):
                    facts = namespace[function_name](
                        "用户没有提供事实",
                        "普通回答",
                        api_client=fake_client(response),
                    )
                    self.assertEqual([], facts)

    def test_fact_distillation_accepts_complete_real_response_shape(self):
        cases = (
            ("d4_3_with_memory.py", "extract_facts_from_conversation"),
            ("d4_4_memory_agent.py", "extract_facts"),
        )
        for file_name, function_name in cases:
            with self.subTest(file_name=file_name):
                namespace = load_module(file_name)
                facts = namespace[function_name](
                    "我是 Python 开发者",
                    "收到",
                    api_client=fake_client(
                        full_response(content='["用户是 Python 开发者"]')
                    ),
                )
                self.assertEqual(["用户是 Python 开发者"], facts)

    def test_memory_chat_rejects_empty_final_answer_before_distillation(self):
        cases = (
            ("d4_3_with_memory.py", "chat_with_memory"),
            ("d4_4_memory_agent.py", "chat"),
        )

        class EmptyCollection:
            def count(self):
                return 0

        for file_name, function_name in cases:
            namespace = load_module(file_name)
            for response in (
                full_response(choices=False),
                full_response(message=False),
                full_response(content=None),
            ):
                with self.subTest(file_name=file_name, response=response):
                    with self.assertRaisesRegex(RuntimeError, "空响应"):
                        namespace[function_name](
                            "你好",
                            api_client=fake_client(response),
                            collection=EmptyCollection(),
                        )


class AccessCountTests(unittest.TestCase):
    class StatefulCollection:
        def __init__(self):
            self.records = {
                "memory-1": {
                    "document": "用户是 Python 开发者",
                    "metadata": {
                        "created_at": time.time(),
                        "access_count": 0,
                    },
                }
            }
            self.updated_ids = []

        def count(self):
            return len(self.records)

        def query(self, **_kwargs):
            record = self.records["memory-1"]
            return {
                "ids": [["memory-1"]],
                "documents": [[record["document"]]],
                "metadatas": [[dict(record["metadata"])]],
                "distances": [[0.01]],
            }

        def update(self, *, ids, metadatas):
            self.updated_ids.append(list(ids))
            for memory_id, metadata in zip(ids, metadatas):
                self.records[memory_id]["metadata"] = dict(metadata)

    def test_repeated_recall_increments_returned_ids(self):
        cases = (
            ("d4_3_with_memory.py", "search_memory"),
            ("d4_4_memory_agent.py", "recall_memory"),
        )
        for file_name, function_name in cases:
            with self.subTest(file_name=file_name):
                namespace = load_module(file_name)
                collection = self.StatefulCollection()

                first = namespace[function_name](
                    "Python",
                    collection=collection,
                )
                self.assertEqual(["用户是 Python 开发者"], first)
                self.assertEqual(
                    1,
                    collection.records["memory-1"]["metadata"]["access_count"],
                )

                second = namespace[function_name](
                    "Python",
                    collection=collection,
                )
                self.assertEqual(["用户是 Python 开发者"], second)
                self.assertEqual(
                    2,
                    collection.records["memory-1"]["metadata"]["access_count"],
                )
                self.assertEqual(
                    [["memory-1"], ["memory-1"]],
                    collection.updated_ids,
                )


class ResourceLifecycleTests(unittest.TestCase):
    def test_memory_examples_explicitly_close_owned_database(self):
        class FakeDatabase:
            def __init__(self, *, persistent):
                self.persistent = persistent
                self.closed = False
                self.collection = SimpleNamespace(count=lambda: 0)

            def has_collection(self, _name):
                return self.persistent

            def delete_collection(self, _name):
                return None

            def create_collection(self, name):
                return self.collection

            def get_collection(self, name):
                return self.collection

            def __exit__(self, exc_type, exc_value, traceback):
                self.closed = True

        for file_name, persistent in (
            ("d4_3_with_memory.py", False),
            ("d4_4_memory_agent.py", True),
        ):
            with self.subTest(file_name=file_name):
                namespace = load_module(file_name)
                database = FakeDatabase(persistent=persistent)
                namespace["initialize_runtime"](
                    api_client=object(),
                    database=database,
                )
                namespace["close_runtime"]()

                self.assertTrue(database.closed)
                runtime_globals = namespace["close_runtime"].__globals__
                self.assertIsNone(runtime_globals["client"])
                self.assertIsNone(runtime_globals["db"])
                self.assertIsNone(runtime_globals["memory_col"])


if __name__ == "__main__":
    unittest.main()
