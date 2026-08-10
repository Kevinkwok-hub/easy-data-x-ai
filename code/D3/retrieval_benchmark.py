"""D3 检索三角基准：统一计算精度、延迟和可解释成本。"""

from __future__ import annotations

import gc
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns
from typing import Callable, Sequence

from rag_engineering import Evidence, QueryAnalysis, analyze_query
from rag_evaluation import EvaluationCase


Retriever = Callable[
    [str, QueryAnalysis],
    Sequence[Sequence[Evidence]],
]


@dataclass(frozen=True)
class CostAssumptions:
    """把易变的供应商单价与可复现的 Token 计数分开。"""

    input_price_per_million_tokens: float = 1.0
    currency: str = "CNY"

    def __post_init__(self) -> None:
        if self.input_price_per_million_tokens < 0:
            raise ValueError("input_price_per_million_tokens 不能为负数")
        if not self.currency.strip():
            raise ValueError("currency 不能为空")


@dataclass(frozen=True)
class BenchmarkConfig:
    """控制预热、采样和成本换算口径。"""

    warmup_rounds: int = 5
    measurement_rounds: int = 30
    top_k: int = 3
    cost: CostAssumptions = CostAssumptions()

    def __post_init__(self) -> None:
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds 不能为负数")
        if self.measurement_rounds <= 0:
            raise ValueError("measurement_rounds 必须大于 0")
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")


@dataclass(frozen=True)
class StrategyBenchmark:
    strategy: str
    label: str
    hit_at_1: float
    hit_at_3: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    average_context_tokens: float
    context_cost_per_1k_queries: float
    database_requests_per_query: int
    embedding_calls_per_query: int
    vector_branches_per_query: int
    fulltext_branches_per_query: int
    fusion_steps_per_query: int


@dataclass(frozen=True)
class BenchmarkReport:
    backend: str
    case_count: int
    samples_per_strategy: int
    environment: str
    config: BenchmarkConfig
    strategies: tuple[StrategyBenchmark, ...]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """使用 nearest-rank 口径，避免依赖第三方统计库。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _flatten(groups: Sequence[Sequence[Evidence]]) -> list[Evidence]:
    return [evidence for group in groups for evidence in group]


def _estimate_tokens(text: str) -> int:
    """沿用 D3 的教学估算：约 3 个字符折算为 1 个 Token。"""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def _relative_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline


def _measure_strategy(
    *,
    strategy: str,
    label: str,
    retriever: Retriever,
    cases: Sequence[EvaluationCase],
    config: BenchmarkConfig,
    fulltext_branches: int,
    fusion_steps: int,
) -> StrategyBenchmark:
    prepared_cases = [(case, analyze_query(case.question)) for case in cases]

    for _ in range(config.warmup_rounds):
        for case, analysis in prepared_cases:
            retriever(case.question, analysis)

    latencies_ms: list[float] = []
    first_round_results: list[list[Evidence]] = []
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for round_index in range(config.measurement_rounds):
            for case, analysis in prepared_cases:
                started_at = perf_counter_ns()
                groups = retriever(case.question, analysis)
                latencies_ms.append((perf_counter_ns() - started_at) / 1_000_000)
                if round_index == 0:
                    first_round_results.append(_flatten(groups))
    finally:
        if gc_was_enabled:
            gc.enable()

    hit_at_1: list[float] = []
    hit_at_3: list[float] = []
    reciprocal_ranks: list[float] = []
    context_tokens: list[int] = []
    for (case, _), evidence in zip(prepared_cases, first_round_results):
        expected = set(case.expected_doc_ids)
        top_k_evidence = evidence[: config.top_k]
        hit_at_1.append(float(bool(evidence and evidence[0].doc_id in expected)))
        hit_at_3.append(
            float(any(item.doc_id in expected for item in evidence[:3]))
        )
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, item in enumerate(evidence, start=1)
                    if item.doc_id in expected
                ),
                0.0,
            )
        )
        context_tokens.append(
            sum(_estimate_tokens(item.content) for item in top_k_evidence)
        )

    average_context_tokens = mean(context_tokens)
    context_cost_per_1k = (
        average_context_tokens
        * config.cost.input_price_per_million_tokens
        / 1_000
    )
    return StrategyBenchmark(
        strategy=strategy,
        label=label,
        hit_at_1=round(mean(hit_at_1), 4),
        hit_at_3=round(mean(hit_at_3), 4),
        mrr=round(mean(reciprocal_ranks), 4),
        latency_p50_ms=round(median(latencies_ms), 4),
        latency_p95_ms=round(_percentile(latencies_ms, 0.95), 4),
        average_context_tokens=round(average_context_tokens, 2),
        context_cost_per_1k_queries=round(context_cost_per_1k, 4),
        database_requests_per_query=1,
        embedding_calls_per_query=1,
        vector_branches_per_query=1,
        fulltext_branches_per_query=fulltext_branches,
        fusion_steps_per_query=fusion_steps,
    )


def run_retrieval_triangle_benchmark(
    cases: Sequence[EvaluationCase],
    *,
    vector_retriever: Retriever,
    hybrid_retriever: Retriever,
    config: BenchmarkConfig = BenchmarkConfig(),
    backend: str = "offline-simulator",
) -> BenchmarkReport:
    """在同一批可回答问题上运行纯向量与混合检索。"""
    answerable_cases = [case for case in cases if case.should_answer]
    if not answerable_cases:
        raise ValueError("基准至少需要一条可回答案例")

    strategies = (
        _measure_strategy(
            strategy="vector",
            label="纯向量",
            retriever=vector_retriever,
            cases=answerable_cases,
            config=config,
            fulltext_branches=0,
            fusion_steps=0,
        ),
        _measure_strategy(
            strategy="hybrid",
            label="混合检索",
            retriever=hybrid_retriever,
            cases=answerable_cases,
            config=config,
            fulltext_branches=1,
            fusion_steps=1,
        ),
    )
    return BenchmarkReport(
        backend=backend,
        case_count=len(answerable_cases),
        samples_per_strategy=len(answerable_cases) * config.measurement_rounds,
        environment=(
            f"{platform.system()} {platform.machine()} / "
            f"Python {platform.python_version()}"
        ),
        config=config,
        strategies=strategies,
    )


def render_markdown(report: BenchmarkReport) -> str:
    """生成可审阅、可复算的三角对比报告。"""
    by_name = {result.strategy: result for result in report.strategies}
    vector = by_name["vector"]
    hybrid = by_name["hybrid"]
    currency = report.config.cost.currency
    price = report.config.cost.input_price_per_million_tokens

    hit_at_1_delta_pp = (hybrid.hit_at_1 - vector.hit_at_1) * 100
    latency_delta = _relative_change(
        hybrid.latency_p95_ms,
        vector.latency_p95_ms,
    )
    token_delta = _relative_change(
        hybrid.average_context_tokens,
        vector.average_context_tokens,
    )

    is_offline = report.backend == "offline-simulator"
    scope_notice = (
        "> 这是不需要数据库和 API Key 的教学基准。它使用固定离线检索器隔离检索策略差异，\n"
        "> 不应被当作 seekdb、Embedding 服务或线上硬件的性能承诺。"
        if is_offline
        else
        "> 这是当前 seekdb 集合的实测结果。它仍只代表本次数据量、索引、机器和并发条件，\n"
        "> 不应被外推为其他部署环境的性能承诺。"
    )
    lines = [
        "# D3 混合检索 vs 纯向量：延迟 / 成本 / 精度三角实验",
        "",
        *scope_notice.splitlines(),
        "",
        "## 实验设置",
        "",
        f"- 检索后端：`{report.backend}`。",
        f"- 可回答案例：{report.case_count} 条；两种策略使用完全相同的问题与 Top-{report.config.top_k}。",
        f"- 预热：{report.config.warmup_rounds} 轮；正式采样：{report.config.measurement_rounds} 轮，"
        f"每种策略共 {report.samples_per_strategy} 次检索。",
        "- 延迟：只包围检索函数，不含数据加载、查询分析、答案生成和报告写入。",
        "- 精度：Hit@1、Hit@3 与 MRR 只在有标注答案的案例上计算，避免把安全拒答逻辑混进检索质量。",
        f"- 成本示例：输入上下文按 {currency} {price:g} / 百万 Token 换算；该单价仅用于展示公式，可通过命令行替换。",
        f"- 运行环境：{report.environment}。",
        "",
        "## 三角对比",
        "",
        "| 方案 | Hit@1 | Hit@3 | MRR | P50（ms） | P95（ms） | 平均上下文 Token / 查询 | 示例上下文成本 / 千次查询 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.strategies:
        lines.append(
            f"| {result.label} | {result.hit_at_1:.2f} | "
            f"{result.hit_at_3:.2f} | {result.mrr:.2f} | "
            f"{result.latency_p50_ms:.4f} | {result.latency_p95_ms:.4f} | "
            f"{result.average_context_tokens:.2f} | {currency} "
            f"{result.context_cost_per_1k_queries:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 相对纯向量的变化",
            "",
            "| 精度收益 | 延迟代价 | 上下文成本代价 |",
            "| ---: | ---: | ---: |",
            f"| Hit@1 {hit_at_1_delta_pp:+.1f} 个百分点 | "
            f"P95 {latency_delta:+.1%} | Token / 查询 {token_delta:+.1%} |",
            "",
            "## 成本拆解",
            "",
            "| 方案 | 数据库请求 / 查询 | 查询 Embedding / 查询 | 向量分支 | 全文分支 | RRF 融合 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report.strategies:
        lines.append(
            f"| {result.label} | {result.database_requests_per_query} | "
            f"{result.embedding_calls_per_query} | "
            f"{result.vector_branches_per_query} | "
            f"{result.fulltext_branches_per_query} | "
            f"{result.fusion_steps_per_query} |"
        )
    lines.extend(
        [
            "",
            (
                "离线模式不会真的发出数据库或 Embedding 请求；上表描述的是把同一策略替换为 seekdb 后的等价线上调用拓扑。"
                if is_offline
                else
                "下表描述本次 seekdb 检索的应用侧请求与引擎内部查询分支。"
            ),
            "",
            "seekdb 的混合检索由一次数据库请求完成，应用侧不需要发两次请求；额外工作发生在引擎内部的全文分支与 RRF 融合。两种策略都只需要一次查询 Embedding。",
            "",
            "上下文成本按下面的公式换算：",
            "",
            "```text",
            "千次查询上下文成本 = 平均上下文 Token / 查询 × 1000",
            "                     ÷ 1,000,000 × 每百万输入 Token 单价",
            "```",
            "",
            "该成本不含数据库实例、索引存储、Embedding 服务和答案输出 Token。生产评估时应把同一脚本的 Token 结果乘以实际模型价格，并用数据库监控补充 CPU、内存和存储账单。",
            "",
            "## 选型结论",
            "",
            "- 错误码、版本号、函数名等精确标识符占比高，或答错代价高：优先混合检索。",
            "- 查询几乎都是语义改写且延迟预算极紧：可保留纯向量基线，再用业务评测集验证精度是否达标。",
            "- 更常见的生产选择是自适应路由：普通语义问题走向量，检测到精确标识符时走混合检索。",
            (
                "- 上线前必须在目标机器、真实索引规模和并发量下重新测 P95 / P99；本报告中的毫秒值只用于比较本机两种离线策略。"
                if is_offline
                else
                "- 上线前还应使用真实索引规模和并发压测 P95 / P99，并连续记录数据库 CPU、内存和存储账单。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(report: BenchmarkReport, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8")
