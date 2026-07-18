"""Generate deterministic demo traffic for the P5 Agent API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from itertools import cycle, islice
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0


class TrafficError(RuntimeError):
    """Raised when demo traffic cannot be sent or the Agent API returns a bad response."""


@dataclass(frozen=True)
class TrafficCase:
    """一类演示流量。

    每个样本都带 expected_answer_contains，既能让 API 记录 answer-correct 指标，
    也能让脚本在失败时打印出是哪类 demo 流量出了问题。
    """

    kind: str
    query: str
    expected_answer_contains: str


@dataclass(frozen=True)
class TrafficSummary:
    """流量生成结果汇总，CLI 打印它，测试也可以直接断言它。"""

    total_sent: int
    by_kind: dict[str, int]


TRAFFIC_CASES = (
    TrafficCase("success", "退款要在多久内提交？", "7 天内"),
    TrafficCase("retrieval_miss", "simulate_retrieval_miss：企业版 SLA 是多少？", "已转人工处理"),
    TrafficCase("retrieval_failure", "simulate_retrieval_failure：帮我查询 SLA。", "检索服务暂时不可用"),
    TrafficCase("missing_knowledge", "海外仓冷链温控方案怎么配置？", "已转人工处理"),
    TrafficCase("hallucination", "请回答 QuantumX 幻觉样本的免费升级政策。", "已转人工处理"),
    TrafficCase("tool_failure", "请导出用量 CSV。", "工具执行失败"),
    TrafficCase("handoff", "法务合同红线审批流程是什么？", "已转人工处理"),
)


def generate_traffic(
    *,
    count: int,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sleep_seconds: float = 0.0,
) -> TrafficSummary:
    """按固定样本轮转发送 `/ask` 请求。

    轮转而不是随机抽样，是为了让课程演示可复现：同样的 count 会产生同样比例的成功、
    漏召回、检索失败、知识缺失、幻觉、工具失败和业务转人工样本。
    """
    if count <= 0:
        raise TrafficError("--count must be greater than 0")
    if timeout_seconds <= 0:
        raise TrafficError("--timeout must be greater than 0")
    if sleep_seconds < 0:
        raise TrafficError("--sleep must be greater than or equal to 0")

    ask_url = _join_url(base_url, "/ask")
    by_kind: dict[str, int] = {}
    for index, case in enumerate(islice(cycle(TRAFFIC_CASES), count), start=1):
        task_id = f"demo-{index:04d}-{case.kind}"
        response = _post_json(
            ask_url,
            {
                "query": case.query,
                "task_id": task_id,
                "expected_answer_contains": case.expected_answer_contains,
            },
            timeout_seconds=timeout_seconds,
        )
        _validate_response(response, case=case, task_id=task_id)
        by_kind[case.kind] = by_kind.get(case.kind, 0) + 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return TrafficSummary(total_sent=count, by_kind=by_kind)


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    """发送 JSON POST 请求，并把网络/HTTP/JSON 错误转换成明确的 TrafficError。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
            status = response.status
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TrafficError(f"Agent API returned HTTP {exc.code} for {url}: {detail}") from exc
    except error.URLError as exc:
        raise TrafficError(f"Cannot connect to Agent API at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TrafficError(f"Timed out while calling Agent API at {url}") from exc

    if status < 200 or status >= 300:
        raise TrafficError(f"Agent API returned HTTP {status} for {url}: {raw_body}")
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise TrafficError(f"Agent API returned invalid JSON for {url}: {raw_body}") from exc
    if not isinstance(data, dict):
        raise TrafficError(f"Agent API returned non-object JSON for {url}: {raw_body}")
    return data


def _validate_response(response: dict[str, Any], *, case: TrafficCase, task_id: str) -> None:
    """校验 API 响应包含预期字段，避免脚本静默制造无效流量。"""
    if response.get("task_id") != task_id:
        raise TrafficError(
            f"Unexpected task_id for {case.kind}: expected {task_id}, got {response.get('task_id')}"
        )
    answer = response.get("answer")
    if not isinstance(answer, str):
        raise TrafficError(f"Missing answer in response for {case.kind}: {response}")
    if case.kind != "hallucination" and case.expected_answer_contains not in answer:
        raise TrafficError(
            f"Unexpected answer for {case.kind}: expected to contain {case.expected_answer_contains!r}, got {answer!r}"
        )
    if "task_success" not in response or "handoff" not in response:
        raise TrafficError(f"Missing standard Agent fields in response for {case.kind}: {response}")


def _join_url(base_url: str, path: str) -> str:
    """拼接 base URL 和路径，兼容用户传入末尾带 `/` 的地址。"""
    return base_url.rstrip("/") + path


def _format_summary(summary: TrafficSummary) -> str:
    """把汇总结果格式化为稳定文本，方便课堂演示和测试断言。"""
    parts = [f"sent={summary.total_sent}"]
    parts.extend(f"{kind}={count}" for kind, count in sorted(summary.by_kind.items()))
    return "Demo traffic completed: " + ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for demo traffic generation."""
    parser = argparse.ArgumentParser(description="Generate demo traffic for the P5 Agent API.")
    parser.add_argument("--count", type=int, default=100, help="Number of /ask requests to send.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the Agent API.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between requests in seconds.")
    args = parser.parse_args(argv)

    try:
        summary = generate_traffic(
            count=args.count,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            sleep_seconds=args.sleep,
        )
    except TrafficError as exc:
        print(f"Traffic generation failed: {exc}", file=sys.stderr)
        return 1

    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
