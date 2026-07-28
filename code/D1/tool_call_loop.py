import json

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.tools import BaseTool


_REQUIRED_RESPONSE_FIELDS = ("content", "tool_calls", "invalid_tool_calls")


def _require_response_fields(response):
    """检查每个流式消息块是否具备工具循环所需的标准字段。"""
    if response is None:
        raise RuntimeError("模型返回了空响应")
    missing = [
        field
        for field in _REQUIRED_RESPONSE_FIELDS
        if not hasattr(response, field)
    ]
    if missing:
        raise RuntimeError(
            f"模型响应缺少必要字段：{', '.join(missing)}"
        )
    return response


def aggregate_stream(chunks):
    """增量聚合真实 LangChain 消息块，并拒绝结构无效的流。"""
    iterator = iter(chunks)
    try:
        response = _require_response_fields(next(iterator))
    except StopIteration:
        raise RuntimeError("模型未返回任何响应") from None

    for chunk in iterator:
        valid_chunk = _require_response_fields(chunk)
        try:
            response = response + valid_chunk
        except Exception as exc:
            raise RuntimeError(f"模型流式响应聚合失败：{exc}") from exc
        _require_response_fields(response)
    return response


def _has_nonempty_content(content):
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            bool(item.strip()) if isinstance(item, str) else bool(item)
            for item in content
        )
    return False


def _validate_tool_calls(
    response,
    tools_by_name,
    *,
    require_tool_call_ids,
):
    """使用聚合后的原始 JSON 参数整批校验工具调用。"""
    invalid_tool_calls = response.invalid_tool_calls
    if invalid_tool_calls:
        raise RuntimeError(
            f"模型返回了无效工具调用 invalid_tool_calls：{invalid_tool_calls}"
        )

    tool_calls = response.tool_calls
    if not isinstance(tool_calls, list):
        raise RuntimeError("模型响应的 tool_calls 必须为列表")
    if not tool_calls:
        return []

    raw_chunks = getattr(response, "tool_call_chunks", None)
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(tool_calls):
        raise RuntimeError("模型响应缺少完整的 tool_call_chunks 原始参数")

    validated_calls = []
    seen_ids = set()
    for index, (tool_call, raw_chunk) in enumerate(
        zip(tool_calls, raw_chunks),
        start=1,
    ):
        if not isinstance(tool_call, dict) or not isinstance(raw_chunk, dict):
            raise RuntimeError(f"第 {index} 个工具调用结构无效：必须为字典")

        tool_name = tool_call.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise RuntimeError(f"第 {index} 个工具调用的 name 必须为非空字符串")
        raw_name = raw_chunk.get("name")
        if raw_name not in (None, tool_name):
            raise RuntimeError(f"第 {index} 个工具调用名称与原始分片不一致")
        if tool_name not in tools_by_name:
            raise RuntimeError(f"模型请求了未注册的工具：{tool_name}")

        raw_args = raw_chunk.get("args")
        if not isinstance(raw_args, str):
            raise RuntimeError(f"工具 {tool_name} 的原始 args 必须是 JSON 字符串")
        try:
            tool_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"工具 {tool_name} 的原始 args 不是完整合法 JSON"
            ) from exc
        if not isinstance(tool_args, dict):
            raise RuntimeError(f"工具 {tool_name} 的 args 必须为 JSON 对象")

        tool_call_id = tool_call.get("id")
        raw_id = raw_chunk.get("id")
        if raw_id not in (None, tool_call_id):
            raise RuntimeError(f"第 {index} 个工具调用 ID 与原始分片不一致")
        if require_tool_call_ids:
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise RuntimeError(
                    f"标准 LangChain 协议要求第 {index} 个工具调用具有非空 ID"
                )
            if tool_call_id in seen_ids:
                raise RuntimeError(
                    f"标准 LangChain 协议要求工具调用 ID 唯一：{tool_call_id}"
                )
            seen_ids.add(tool_call_id)

        validated_call = dict(tool_call)
        validated_call["args"] = tool_args
        validated_call["type"] = "tool_call"
        validated_calls.append(
            (validated_call, tools_by_name[tool_name])
        )
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


def _tool_message_content(result):
    """把常见工具返回值转换成 ToolMessage 可接受的内容。"""
    if isinstance(result, (str, list)):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _standard_assistant_message(response):
    try:
        assistant_message = message_chunk_to_message(response)
    except Exception as exc:
        raise RuntimeError(f"无法把聚合响应转换为 assistant 消息：{exc}") from exc
    if not isinstance(assistant_message, AIMessage):
        raise RuntimeError("聚合响应无法转换为标准 AIMessage")
    return assistant_message


def _standard_tool_message(tool_call, result):
    if isinstance(result, ToolMessage):
        if result.tool_call_id != tool_call["id"]:
            raise RuntimeError(
                f"工具 {tool_call['name']} 返回的 ToolMessage ID 不匹配"
            )
        return result
    return ToolMessage(
        content=_tool_message_content(result),
        tool_call_id=tool_call["id"],
    )


def _standard_error_message(tool_call, detail):
    return ToolMessage(
        content=detail,
        tool_call_id=tool_call["id"],
        status="error",
    )


def _append_skipped_tool_messages(messages, remaining_calls, reason):
    for tool_call, _tool in remaining_calls:
        messages.append(
            _standard_error_message(
                tool_call,
                f"未执行工具 {tool_call['name']}：{reason}",
            )
        )


def _callback_result(result):
    if isinstance(result, ToolMessage):
        return result.content
    return result


def _tool_input(tool, tool_call, *, legacy_user_message_fallback):
    if not legacy_user_message_fallback and isinstance(tool, BaseTool):
        return tool_call
    return tool_call["args"]


def run_tool_call_loop(
    model,
    messages,
    tools_by_name,
    max_rounds=5,
    on_tool_result=None,
    *,
    legacy_user_message_fallback=False,
):
    """执行多轮工具调用；默认使用标准 LangChain assistant/tool 协议。

    max_rounds 表示最多执行多少轮工具。执行完最后一轮后，模型仍会获得一次
    生成最终答案的机会；只有这次仍请求工具时才报告达到上限。
    """
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds <= 0
    ):
        raise ValueError("max_rounds 必须是正整数")

    executed_tool_rounds = 0
    while True:
        response = aggregate_stream(model.stream(messages))
        validated_calls = _validate_tool_calls(
            response,
            tools_by_name,
            require_tool_call_ids=not legacy_user_message_fallback,
        )
        if not validated_calls:
            if not _has_nonempty_content(response.content):
                raise RuntimeError("模型未返回工具调用，且最终回答为空")
            return response

        if executed_tool_rounds >= max_rounds:
            raise RuntimeError(
                f"工具调用已达到 {max_rounds} 轮上限，"
                "最终生成机会仍请求了工具"
            )

        if not legacy_user_message_fallback:
            messages.append(_standard_assistant_message(response))

        for index, (tool_call, tool) in enumerate(validated_calls, start=1):
            tool_input = _tool_input(
                tool,
                tool_call,
                legacy_user_message_fallback=legacy_user_message_fallback,
            )
            try:
                result = tool.invoke(tool_input)
            except Exception as exc:
                error_detail = f"工具 {tool_call['name']} 执行失败"
                if legacy_user_message_fallback:
                    messages.append(
                        (
                            "user",
                            _format_tool_result_message(
                                tool_call,
                                error_detail,
                            ),
                        )
                    )
                else:
                    messages.append(
                        _standard_error_message(tool_call, error_detail)
                    )
                    _append_skipped_tool_messages(
                        messages,
                        validated_calls[index:],
                        "同一批次中的前序工具执行失败",
                    )
                raise RuntimeError(
                    f"第 {index} 个工具 {tool_call['name']} 执行失败：{exc}"
                ) from exc

            if legacy_user_message_fallback:
                messages.append(
                    (
                        "user",
                        _format_tool_result_message(tool_call, result),
                    )
                )
            else:
                try:
                    messages.append(
                        _standard_tool_message(tool_call, result)
                    )
                except Exception as exc:
                    error_detail = f"工具 {tool_call['name']} 的结果消息无效"
                    messages.append(
                        _standard_error_message(tool_call, error_detail)
                    )
                    _append_skipped_tool_messages(
                        messages,
                        validated_calls[index:],
                        "同一批次中的前序工具结果消息无效",
                    )
                    raise RuntimeError(
                        f"第 {index} 个工具 {tool_call['name']} "
                        f"的结果消息无效：{exc}"
                    ) from exc

            if on_tool_result is not None:
                try:
                    on_tool_result(tool_call, _callback_result(result))
                except Exception as exc:
                    if not legacy_user_message_fallback:
                        _append_skipped_tool_messages(
                            messages,
                            validated_calls[index:],
                            "同一批次中的结果回调失败",
                        )
                    raise RuntimeError(
                        f"第 {index} 个工具 {tool_call['name']} "
                        f"的结果回调失败：{exc}"
                    ) from exc

        executed_tool_rounds += 1
