"""FastAPI entry point for the P5 offline Knowledge Agent demo."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST

from app.agent import KnowledgeAgent
from app.observability.langsmith_tracing import LangSmithTracer
from app.observability.prometheus_metrics import metrics


app = FastAPI(title="P5 Knowledge Agent Demo", version="0.3.0")
# Agent、Prometheus metrics 和 LangSmith tracer 都在进程启动时初始化；
# 这样课堂演示时每次请求共享同一套本地知识库和指标累计状态。
agent = KnowledgeAgent()
tracer = LangSmithTracer.from_env()


class AskRequest(BaseModel):
    """`POST /ask` 的请求体。

    expected_answer_contains 是教程阶段的可选评测字段，仅用于记录
    agent_answer_evaluated_total；命中关键事实时再记录 agent_answer_correct_total。
    该字段仅用于演示，生产准确率应来自抽样人工标注或独立 Judge。
    """

    query: str = Field(min_length=1, description="User question sent to the Knowledge Agent.")
    task_id: str | None = Field(default=None, description="Optional caller-provided task identifier.")
    expected_answer_contains: str | None = Field(
        default=None,
        description="Optional key fact used to increment answer-correct metrics in demos.",
    )


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查只返回稳定状态字段，方便 Docker、Prometheus 或脚本探活。"""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """运行一次 Knowledge Agent，并把结果同步记录到 Prometheus 指标。"""
    with metrics.track_request():
        result = tracer.trace_agent_run(
            query=request.query,
            task_id=request.task_id,
            expected_answer_contains=request.expected_answer_contains,
            run_agent=lambda: agent.run(request.query, task_id=request.task_id),
        )
        metrics.record_agent_result(result, expected_answer_contains=request.expected_answer_contains)
        payload = result.to_dict()
        payload["task_id"] = request.task_id
        return payload


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """返回 Prometheus exposition 格式，供 Prometheus server 定时抓取。"""
    return Response(content=metrics.render_latest(), media_type=CONTENT_TYPE_LATEST)
