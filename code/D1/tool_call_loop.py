def aggregate_stream(chunks):
    """增量聚合模型返回的流式响应。"""
    iterator = iter(chunks)
    try:
        response = next(iterator)
    except StopIteration:
        raise RuntimeError("模型未返回任何响应") from None

    for chunk in iterator:
        response += chunk
    return response


def _validate_tool_calls(tool_calls, tools_by_name):
    validated_calls = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            raise RuntimeError(f"第 {index} 个工具调用结构无效：必须为字典")

        tool_name = tool_call.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise RuntimeError(f"第 {index} 个工具调用的 name 必须为非空字符串")

        tool_args = tool_call.get("args")
        if not isinstance(tool_args, dict):
            raise RuntimeError(f"工具 {tool_name} 的 args 必须为字典")

        if tool_name not in tools_by_name:
            raise RuntimeError(f"模型请求了未注册的工具：{tool_name}")

        validated_calls.append((tool_call, tools_by_name[tool_name]))
    return validated_calls


def _format_tool_result_message(tool_call, result):
    lines = [
        f"工具名称：{tool_call['name']}",
        f"调用参数：{tool_call['args']!r}",
    ]
    if tool_call.get("id") is not None:
        lines.append(f"调用 ID：{tool_call['id']}")
    lines.append(f"执行结果：\n\n{result}")
    return "\n".join(lines)


def run_tool_call_loop(
    model,
    messages,
    tools_by_name,
    max_rounds=5,
    on_tool_result=None,
):
    """循环执行全部工具调用，并原地向 messages 追加每个工具结果。"""
    for _ in range(max_rounds):
        response = aggregate_stream(model.stream(messages))
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response

        validated_calls = _validate_tool_calls(tool_calls, tools_by_name)
        for tool_call, tool in validated_calls:
            result = tool.invoke(tool_call["args"])
            messages.append(
                (
                    "user",
                    _format_tool_result_message(tool_call, result),
                )
            )
            if on_tool_result is not None:
                on_tool_result(tool_call, result)

    raise RuntimeError(f"工具调用已达到 {max_rounds} 轮上限，仍未得到最终回答")
