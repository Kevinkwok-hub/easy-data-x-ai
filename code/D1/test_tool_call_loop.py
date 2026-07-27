import unittest

from tool_call_loop import aggregate_stream, run_tool_call_loop


class FakeChunk:
    def __init__(self, content="", tool_calls=None, state=None):
        self.content = content
        self.tool_calls = list(tool_calls or [])
        self.state = state

    def __add__(self, other):
        if self.state is not None:
            self.state["aggregations"] += 1
        return FakeChunk(
            content=self.content + other.content,
            tool_calls=self.tool_calls + other.tool_calls,
            state=self.state,
        )


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream(self, messages):
        self.calls.append(list(messages))
        chunks = self.responses.pop(0)
        return (chunk for chunk in chunks)


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


class AggregateStreamTest(unittest.TestCase):
    def test_aggregate_stream_consumes_and_aggregates_incrementally(self):
        state = {"aggregations": 0}

        def chunks():
            yield FakeChunk("A", state=state)
            yield FakeChunk("B", state=state)
            self.assertEqual(state["aggregations"], 1)
            yield FakeChunk("C", state=state)

        response = aggregate_stream(chunks())

        self.assertEqual(response.content, "ABC")
        self.assertEqual(state["aggregations"], 2)

    def test_aggregate_stream_rejects_empty_stream(self):
        with self.assertRaisesRegex(RuntimeError, "模型未返回任何响应"):
            aggregate_stream(iter(()))


class RunToolCallLoopTest(unittest.TestCase):
    def test_executes_every_tool_call_in_the_same_round(self):
        tool_a = FakeTool("A 结果")
        tool_b = FakeTool("B 结果")
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "tool_a", "args": {"value": 1}},
                            {"name": "tool_b", "args": {"value": 2}},
                        ]
                    )
                ],
                [FakeChunk(content="最终回答")],
            ]
        )
        callbacks = []
        messages = [("user", "问题")]

        response = run_tool_call_loop(
            model,
            messages,
            {"tool_a": tool_a, "tool_b": tool_b},
            on_tool_result=lambda tool_call, result: callbacks.append(
                (tool_call, result)
            ),
        )

        self.assertEqual(response.content, "最终回答")
        self.assertEqual(tool_a.calls, [{"value": 1}])
        self.assertEqual(tool_b.calls, [{"value": 2}])
        self.assertEqual(
            callbacks,
            [
                ({"name": "tool_a", "args": {"value": 1}}, "A 结果"),
                ({"name": "tool_b", "args": {"value": 2}}, "B 结果"),
            ],
        )
        self.assertEqual(len(model.calls), 2)
        self.assertIn("A 结果", model.calls[1][-2][1])
        self.assertIn("B 结果", model.calls[1][-1][1])

    def test_same_tool_calls_include_their_own_args_and_ids_in_messages(self):
        tool = FakeTool("查询结果")
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {
                                "id": "call_first",
                                "name": "search",
                                "args": {"query": "第一个问题"},
                            },
                            {
                                "id": "call_second",
                                "name": "search",
                                "args": {"query": "第二个问题"},
                            },
                        ]
                    )
                ],
                [FakeChunk(content="最终回答")],
            ]
        )

        run_tool_call_loop(
            model,
            [("user", "问题")],
            {"search": tool},
        )

        first_message = model.calls[1][-2][1]
        second_message = model.calls[1][-1][1]
        self.assertIn("call_first", first_message)
        self.assertIn("'query': '第一个问题'", first_message)
        self.assertNotIn("call_second", first_message)
        self.assertIn("call_second", second_message)
        self.assertIn("'query': '第二个问题'", second_message)
        self.assertNotIn("call_first", second_message)
        self.assertEqual(
            tool.calls,
            [{"query": "第一个问题"}, {"query": "第二个问题"}],
        )

    def test_supports_different_tools_across_multiple_rounds(self):
        tool_a = FakeTool("第一步")
        tool_b = FakeTool("第二步")
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "tool_a", "args": {"query": "先查 A"}}
                        ]
                    )
                ],
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "tool_b", "args": {"query": "再查 B"}}
                        ]
                    )
                ],
                [FakeChunk(content="综合回答")],
            ]
        )

        response = run_tool_call_loop(
            model,
            [("user", "问题")],
            {"tool_a": tool_a, "tool_b": tool_b},
        )

        self.assertEqual(response.content, "综合回答")
        self.assertEqual(tool_a.calls, [{"query": "先查 A"}])
        self.assertEqual(tool_b.calls, [{"query": "再查 B"}])
        self.assertEqual(len(model.calls), 3)
        self.assertIn("第一步", model.calls[1][-1][1])
        self.assertIn("第二步", model.calls[2][-1][1])

    def test_preserves_single_tool_call_behavior(self):
        tool = FakeTool("检索结果")
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "search", "args": {"query": "seekdb"}}
                        ]
                    )
                ],
                [FakeChunk(content="基于结果的回答")],
            ]
        )

        response = run_tool_call_loop(
            model,
            [("user", "问题")],
            {"search": tool},
        )

        self.assertEqual(response.content, "基于结果的回答")
        self.assertEqual(tool.calls, [{"query": "seekdb"}])
        self.assertEqual(len(model.calls), 2)

    def test_returns_immediately_when_model_does_not_call_a_tool(self):
        direct_response = FakeChunk(content="直接回答")
        model = FakeModel([[direct_response]])
        messages = [("user", "简单问题")]

        response = run_tool_call_loop(model, messages, {})

        self.assertIs(response, direct_response)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(messages, [("user", "简单问题")])

    def test_raises_after_maximum_tool_call_rounds(self):
        tool = FakeTool("仍需继续")
        tool_call = {"name": "search", "args": {"query": "继续"}}
        model = FakeModel(
            [
                [FakeChunk(tool_calls=[tool_call])],
                [FakeChunk(tool_calls=[tool_call])],
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "2.*轮"):
            run_tool_call_loop(
                model,
                [("user", "问题")],
                {"search": tool},
                max_rounds=2,
            )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(tool.calls), 2)

    def test_unknown_tool_error_includes_tool_name(self):
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "missing_tool", "args": {"value": 1}}
                        ]
                    )
                ]
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "missing_tool"):
            run_tool_call_loop(model, [("user", "问题")], {})

    def test_mixed_known_and_unknown_tools_fail_before_any_invoke(self):
        known_tool = FakeTool("不应执行")
        model = FakeModel(
            [
                [
                    FakeChunk(
                        tool_calls=[
                            {"name": "known_tool", "args": {"value": 1}},
                            {"name": "missing_tool", "args": {"value": 2}},
                        ]
                    )
                ]
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "missing_tool"):
            run_tool_call_loop(
                model,
                [("user", "问题")],
                {"known_tool": known_tool},
            )

        self.assertEqual(known_tool.calls, [])

    def test_malformed_batch_fails_before_any_invoke(self):
        malformed_calls = [
            None,
            {},
            {"name": "", "args": {}},
            {"name": 123, "args": {}},
            {"name": "known_tool"},
            {"name": "known_tool", "args": []},
        ]

        for malformed_call in malformed_calls:
            with self.subTest(tool_call=malformed_call):
                known_tool = FakeTool("不应执行")
                model = FakeModel(
                    [
                        [
                            FakeChunk(
                                tool_calls=[
                                    {
                                        "name": "known_tool",
                                        "args": {"value": 1},
                                    },
                                    malformed_call,
                                ]
                            )
                        ]
                    ]
                )

                with self.assertRaises(RuntimeError):
                    run_tool_call_loop(
                        model,
                        [("user", "问题")],
                        {"known_tool": known_tool},
                    )

                self.assertEqual(known_tool.calls, [])


if __name__ == "__main__":
    unittest.main()
