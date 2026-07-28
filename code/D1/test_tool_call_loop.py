import unittest
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool

from tool_call_loop import aggregate_stream, run_tool_call_loop


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("模型调用次数超出测试预期")
        return iter(self.responses.pop(0))


class FakeTool:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.invocations = []

    def invoke(self, args):
        self.invocations.append(args)
        if isinstance(args, dict) and args.get("type") == "tool_call":
            self.calls.append(args["args"])
        else:
            self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result


class ArgsOnlyTool:
    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return args["query"]


def final_chunks(content="最终回答"):
    return [AIMessageChunk(content=content)]


def complete_tool_chunks(*calls):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": call["name"],
                    "args": call["raw_args"],
                    "id": call.get("id"),
                    "index": index,
                }
                for index, call in enumerate(calls)
            ],
        )
    ]


def split_two_tool_chunks():
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "tool_a",
                    "args": '{"value":"A',
                    "id": "call_a",
                    "index": 0,
                },
                {
                    "name": "tool_b",
                    "args": '{"value":"B',
                    "id": "call_b",
                    "index": 1,
                },
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": None, "args": '"}', "id": None, "index": 0},
                {"name": None, "args": '"}', "id": None, "index": 1},
            ],
        ),
    ]


class MissingFieldsChunk:
    content = "缺少工具字段"

    def __add__(self, other):
        return self


class EmptyContentChunk:
    content = None
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    def __add__(self, other):
        return self


class AggregateStreamTest(unittest.TestCase):
    def test_aggregates_real_content_chunks_incrementally(self):
        response = aggregate_stream(
            iter(
                (
                    AIMessageChunk(content="A"),
                    AIMessageChunk(content="B"),
                    AIMessageChunk(content="C"),
                )
            )
        )

        self.assertIsInstance(response, AIMessageChunk)
        self.assertEqual(response.content, "ABC")

    def test_aggregates_real_split_tool_call_chunks(self):
        response = aggregate_stream(iter(split_two_tool_chunks()))

        self.assertEqual(
            response.tool_calls,
            [
                {
                    "name": "tool_a",
                    "args": {"value": "A"},
                    "id": "call_a",
                    "type": "tool_call",
                },
                {
                    "name": "tool_b",
                    "args": {"value": "B"},
                    "id": "call_b",
                    "type": "tool_call",
                },
            ],
        )
        self.assertEqual(
            [chunk["args"] for chunk in response.tool_call_chunks],
            ['{"value":"A"}', '{"value":"B"}'],
        )

    def test_rejects_empty_none_or_missing_field_streams(self):
        cases = (
            iter(()),
            iter((None,)),
            iter((MissingFieldsChunk(),)),
        )
        for chunks in cases:
            with self.subTest(chunks=chunks):
                with self.assertRaisesRegex(RuntimeError, "响应|字段"):
                    aggregate_stream(chunks)


class RunToolCallLoopTest(unittest.TestCase):
    def setUp(self):
        self.initial_messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="问题"),
        ]

    def test_standard_protocol_preserves_two_tools_roles_and_ids(self):
        tool_a = FakeTool("A 结果")
        tool_b = FakeTool("B 结果")
        model = FakeModel([split_two_tool_chunks(), final_chunks()])
        messages = list(self.initial_messages)

        response = run_tool_call_loop(
            model,
            messages,
            {"tool_a": tool_a, "tool_b": tool_b},
        )

        self.assertEqual(response.content, "最终回答")
        self.assertEqual(tool_a.calls, [{"value": "A"}])
        self.assertEqual(tool_b.calls, [{"value": "B"}])
        second_round = model.calls[1]
        self.assertEqual(
            [type(message) for message in second_round],
            [SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolMessage],
        )
        assistant = second_round[-3]
        self.assertEqual(
            [call["id"] for call in assistant.tool_calls],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            [message.tool_call_id for message in second_round[-2:]],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            [message.content for message in second_round[-2:]],
            ["A 结果", "B 结果"],
        )

    def test_standard_protocol_supports_multiple_tool_rounds(self):
        tool_a = FakeTool("第一步")
        tool_b = FakeTool("第二步")
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "tool_a",
                        "raw_args": '{"query":"先查 A"}',
                        "id": "call_a",
                    }
                ),
                complete_tool_chunks(
                    {
                        "name": "tool_b",
                        "raw_args": '{"query":"再查 B"}',
                        "id": "call_b",
                    }
                ),
                final_chunks("综合回答"),
            ]
        )
        messages = list(self.initial_messages)

        response = run_tool_call_loop(
            model,
            messages,
            {"tool_a": tool_a, "tool_b": tool_b},
            max_rounds=2,
        )

        self.assertEqual(response.content, "综合回答")
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(
            [type(message) for message in model.calls[2][-4:]],
            [AIMessage, ToolMessage, AIMessage, ToolMessage],
        )
        self.assertEqual(tool_a.calls, [{"query": "先查 A"}])
        self.assertEqual(tool_b.calls, [{"query": "再查 B"}])

    def test_max_rounds_is_a_positive_integer(self):
        for max_rounds in (0, -1, 1.5, True, None):
            with self.subTest(max_rounds=max_rounds):
                model = FakeModel([final_chunks()])
                with self.assertRaisesRegex(ValueError, "max_rounds|正整数"):
                    run_tool_call_loop(
                        model,
                        list(self.initial_messages),
                        {},
                        max_rounds=max_rounds,
                    )
                self.assertEqual(model.calls, [])

    def test_nth_tool_round_gets_one_final_answer_attempt(self):
        tool = FakeTool("结果")
        first_call = complete_tool_chunks(
            {
                "name": "search",
                "raw_args": '{"query":"第一轮"}',
                "id": "call_1",
            }
        )
        second_call = complete_tool_chunks(
            {
                "name": "search",
                "raw_args": '{"query":"第二轮"}',
                "id": "call_2",
            }
        )

        success_model = FakeModel([first_call, second_call, final_chunks("完成")])
        success = run_tool_call_loop(
            success_model,
            list(self.initial_messages),
            {"search": tool},
            max_rounds=2,
        )
        self.assertEqual(success.content, "完成")
        self.assertEqual(len(success_model.calls), 3)
        self.assertEqual(len(tool.calls), 2)

        tool.calls.clear()
        over_limit_model = FakeModel([first_call, second_call, first_call])
        with self.assertRaisesRegex(RuntimeError, "2.*轮"):
            run_tool_call_loop(
                over_limit_model,
                list(self.initial_messages),
                {"search": tool},
                max_rounds=2,
            )
        self.assertEqual(len(over_limit_model.calls), 3)
        self.assertEqual(len(tool.calls), 2)

    def test_rejects_empty_final_content(self):
        for content in (None, "", " \t ", []):
            with self.subTest(content=content):
                chunk = (
                    EmptyContentChunk()
                    if content is None
                    else AIMessageChunk(content=content)
                )
                model = FakeModel([[chunk]])
                with self.assertRaisesRegex(RuntimeError, "最终|空"):
                    run_tool_call_loop(
                        model,
                        list(self.initial_messages),
                        {},
                    )

    def test_rejects_explicit_invalid_tool_calls(self):
        chunk = AIMessageChunk(
            content="",
            invalid_tool_calls=[
                {
                    "name": "search",
                    "args": "{bad json",
                    "id": "call_bad",
                    "error": "invalid json",
                }
            ],
        )
        model = FakeModel([[chunk]])

        with self.assertRaisesRegex(RuntimeError, "invalid_tool_calls|无效工具"):
            run_tool_call_loop(
                model,
                list(self.initial_messages),
                {"search": FakeTool("不应执行")},
            )

    def test_rejects_invalid_or_truncated_raw_json_before_execution(self):
        for raw_args in ("{bad json", '{"query":'):
            with self.subTest(raw_args=raw_args):
                tool = FakeTool("不应执行")
                model = FakeModel(
                    [
                        complete_tool_chunks(
                            {
                                "name": "search",
                                "raw_args": raw_args,
                                "id": "call_bad",
                            }
                        )
                    ]
                )

                with self.assertRaisesRegex(RuntimeError, "JSON"):
                    run_tool_call_loop(
                        model,
                        list(self.initial_messages),
                        {"search": tool},
                    )
                self.assertEqual(tool.calls, [])

    def test_standard_protocol_requires_non_empty_unique_ids(self):
        cases = (
            (
                {
                    "name": "search",
                    "raw_args": '{"query":"A"}',
                    "id": None,
                },
            ),
            (
                {
                    "name": "search",
                    "raw_args": '{"query":"A"}',
                    "id": "same",
                },
                {
                    "name": "search",
                    "raw_args": '{"query":"B"}',
                    "id": "same",
                },
            ),
        )
        for calls in cases:
            with self.subTest(calls=calls):
                tool = FakeTool("不应执行")
                model = FakeModel([complete_tool_chunks(*calls)])
                with self.assertRaisesRegex(RuntimeError, "ID|id|唯一"):
                    run_tool_call_loop(
                        model,
                        list(self.initial_messages),
                        {"search": tool},
                    )
                self.assertEqual(tool.calls, [])

    def test_tool_failure_records_completed_and_failed_calls(self):
        first = FakeTool("第一项已执行")
        second = FakeTool(error=ValueError("数据库失败"))
        third = FakeTool("不应执行")
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "first",
                        "raw_args": '{"value":1}',
                        "id": "call_1",
                    },
                    {
                        "name": "second",
                        "raw_args": '{"value":2}',
                        "id": "call_2",
                    },
                    {
                        "name": "third",
                        "raw_args": '{"value":3}',
                        "id": "call_3",
                    },
                )
            ]
        )
        messages = list(self.initial_messages)
        callbacks = []

        with self.assertRaisesRegex(RuntimeError, "second.*执行失败|执行.*second"):
            run_tool_call_loop(
                model,
                messages,
                {"first": first, "second": second, "third": third},
                on_tool_result=lambda tool_call, result: callbacks.append(
                    (tool_call, result)
                ),
            )

        self.assertEqual(first.invocations[0], {"value": 1})
        self.assertEqual(second.invocations[0], {"value": 2})
        self.assertEqual(third.invocations, [])
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0][0]["id"], "call_1")
        self.assertEqual(callbacks[0][1], "第一项已执行")
        self.assertEqual(
            [type(message) for message in messages[-4:]],
            [AIMessage, ToolMessage, ToolMessage, ToolMessage],
        )
        self.assertEqual(
            [message.status for message in messages[-3:]],
            ["success", "error", "error"],
        )
        self.assertEqual(
            [message.tool_call_id for message in messages[-3:]],
            ["call_1", "call_2", "call_3"],
        )
        self.assertNotIn("数据库失败", messages[-2].content)

    def test_callback_failure_keeps_completed_tool_message(self):
        tool = FakeTool("结果")
        skipped = FakeTool("不应执行")
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "search",
                        "raw_args": '{"query":"A"}',
                        "id": "call_1",
                    },
                    {
                        "name": "skipped",
                        "raw_args": '{"query":"B"}',
                        "id": "call_2",
                    },
                )
            ]
        )
        messages = list(self.initial_messages)

        def failing_callback(_tool_call, _result):
            raise ValueError("显示失败")

        with self.assertRaisesRegex(RuntimeError, "回调.*失败"):
            run_tool_call_loop(
                model,
                messages,
                {"search": tool, "skipped": skipped},
                on_tool_result=failing_callback,
            )

        self.assertEqual(tool.invocations[0], {"query": "A"})
        self.assertEqual(skipped.invocations, [])
        self.assertEqual(
            [type(message) for message in messages[-3:]],
            [AIMessage, ToolMessage, ToolMessage],
        )
        self.assertEqual(
            [message.status for message in messages[-2:]],
            ["success", "error"],
        )
        self.assertEqual(
            [message.tool_call_id for message in messages[-2:]],
            ["call_1", "call_2"],
        )

    def test_standard_protocol_passes_full_call_to_injected_id_tool(self):
        @tool
        def tool_with_id(
            value: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str:
            """返回参数和当前工具调用 ID。"""
            return f"{value}:{tool_call_id}"

        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": tool_with_id.name,
                        "raw_args": '{"value":"A"}',
                        "id": "call_injected",
                    }
                ),
                final_chunks("完成"),
            ]
        )
        messages = list(self.initial_messages)

        response = run_tool_call_loop(
            model,
            messages,
            {tool_with_id.name: tool_with_id},
        )

        self.assertEqual(response.content, "完成")
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertEqual(model.calls[1][-1].content, "A:call_injected")
        self.assertEqual(model.calls[1][-1].tool_call_id, "call_injected")

    def test_standard_protocol_preserves_args_only_custom_tools(self):
        custom_tool = ArgsOnlyTool()
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "custom_search",
                        "raw_args": '{"query":"seekdb"}',
                        "id": "call_custom",
                    }
                ),
                final_chunks("完成"),
            ]
        )

        response = run_tool_call_loop(
            model,
            list(self.initial_messages),
            {"custom_search": custom_tool},
        )

        self.assertEqual(response.content, "完成")
        self.assertEqual(custom_tool.calls, [{"query": "seekdb"}])

    def test_legacy_user_message_fallback_is_explicit(self):
        tool = FakeTool("兼容结果")
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "search",
                        "raw_args": '{"query":"seekdb"}',
                        "id": None,
                    }
                ),
                final_chunks("兼容回答"),
            ]
        )
        messages = list(self.initial_messages)

        response = run_tool_call_loop(
            model,
            messages,
            {"search": tool},
            legacy_user_message_fallback=True,
        )

        self.assertEqual(response.content, "兼容回答")
        self.assertEqual(
            [type(message) for message in model.calls[1]],
            [SystemMessage, HumanMessage, tuple],
        )
        self.assertIn("兼容结果", model.calls[1][-1][1])
        self.assertFalse(
            any(
                isinstance(message, (AIMessage, ToolMessage))
                for message in model.calls[1]
            )
        )

    def test_single_tool_standard_protocol_regression(self):
        tool = FakeTool("检索结果")
        model = FakeModel(
            [
                complete_tool_chunks(
                    {
                        "name": "search",
                        "raw_args": '{"query":"seekdb"}',
                        "id": "call_single",
                    }
                ),
                final_chunks("基于结果的回答"),
            ]
        )
        messages = list(self.initial_messages)

        response = run_tool_call_loop(
            model,
            messages,
            {"search": tool},
        )

        self.assertEqual(response.content, "基于结果的回答")
        self.assertIsInstance(model.calls[1][-2], AIMessage)
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertEqual(model.calls[1][-1].tool_call_id, "call_single")

    def test_unknown_or_malformed_batch_fails_before_any_invoke(self):
        invalid_calls = (
            (
                {
                    "name": "known",
                    "raw_args": '{"value":1}',
                    "id": "call_1",
                },
                {
                    "name": "missing",
                    "raw_args": '{"value":2}',
                    "id": "call_2",
                },
            ),
            (
                {
                    "name": "",
                    "raw_args": '{"value":1}',
                    "id": "call_1",
                },
            ),
            (
                {
                    "name": "known",
                    "raw_args": "[]",
                    "id": "call_1",
                },
            ),
        )

        for calls in invalid_calls:
            with self.subTest(calls=calls):
                known = FakeTool("不应执行")
                model = FakeModel([complete_tool_chunks(*calls)])
                with self.assertRaises(RuntimeError):
                    run_tool_call_loop(
                        model,
                        list(self.initial_messages),
                        {"known": known},
                    )
                self.assertEqual(known.calls, [])

    def test_direct_answer_returns_without_mutating_messages(self):
        model = FakeModel([final_chunks("直接回答")])
        messages = list(self.initial_messages)

        response = run_tool_call_loop(model, messages, {})

        self.assertEqual(response.content, "直接回答")
        self.assertEqual(messages, self.initial_messages)


if __name__ == "__main__":
    unittest.main()
