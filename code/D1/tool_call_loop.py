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


def run_tool_call_loop(
    model,
    messages,
    tools_by_name,
    max_rounds=5,
    on_tool_result=None,
):
    """循环执行模型返回的全部工具调用，直到模型给出最终回答。"""
    for _ in range(max_rounds):
        response = aggregate_stream(model.stream(messages))
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return response

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            if tool_name not in tools_by_name:
                raise RuntimeError(f"模型请求了未注册的工具：{tool_name}")

            result = tools_by_name[tool_name].invoke(tool_call["args"])
            messages.append(
                (
                    "user",
                    f"工具 {tool_name} 的执行结果：\n\n{result}",
                )
            )
            if on_tool_result is not None:
                on_tool_result(tool_call, result)

    raise RuntimeError(f"工具调用已达到 {max_rounds} 轮上限，仍未得到最终回答")
