"""运行 D3 检索三角实验，不需要数据库和 API Key。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

from d3_5_evaluate import hybrid_retrieve, vector_retrieve
from rag_engineering import Evidence, QueryAnalysis
from rag_evaluation import CASES_PATH, load_evaluation_cases
from retrieval_benchmark import (
    BenchmarkConfig,
    CostAssumptions,
    run_retrieval_triangle_benchmark,
    write_markdown_report,
)


REPORT_PATH = Path(__file__).resolve().parent / "reports" / "retrieval-triangle.md"
SEEKDB_REPORT_PATH = (
    Path(__file__).resolve().parent / "reports" / "retrieval-triangle-seekdb.md"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较纯向量与混合检索的精度、延迟和成本",
    )
    parser.add_argument("--warmup", type=int, default=5, help="预热轮数")
    parser.add_argument("--rounds", type=int, default=30, help="正式采样轮数")
    parser.add_argument("--top-k", type=int, default=3, help="注入上下文的结果数")
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=1.0,
        help="每百万输入 Token 的示例单价",
    )
    parser.add_argument("--currency", default="CNY", help="成本展示币种")
    parser.add_argument(
        "--backend",
        choices=("offline", "seekdb"),
        default="offline",
        help="离线教学基准或当前 seekdb 集合实测",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown 报告路径；默认按 backend 使用不同文件名",
    )
    return parser


def _evidence_from_results(results: dict, source: str) -> list[list[Evidence]]:
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    evidence: list[Evidence] = []
    for index, document in enumerate(documents):
        doc_id = ids[index] if index < len(ids) else f"result-{index + 1}"
        distance = distances[index] if index < len(distances) else 0.0
        metadata = metadatas[index] if index < len(metadatas) else {}
        evidence.append(
            Evidence(
                doc_id=str(doc_id),
                content=str(document),
                score=float(distance),
                source=source,
                metadata=metadata or {},
            )
        )
    return [evidence]


def _seekdb_filter(analysis: QueryAnalysis) -> dict[str, str]:
    version = analysis.filters.get("version", "")
    if not version:
        return {}
    return {"version": re.sub(r"^OB-", "", version, flags=re.IGNORECASE)}


def _seekdb_keyword(question: str, analysis: QueryAnalysis) -> str:
    non_version_keywords = [
        keyword
        for keyword in analysis.keywords
        if not re.fullmatch(r"OB-\d+(?:\.\d+)+", keyword, flags=re.IGNORECASE)
    ]
    if non_version_keywords:
        return non_version_keywords[0]
    if analysis.filters:
        return "版本"
    return question.strip().rstrip("？?")


def build_seekdb_retrievers(collection, *, candidate_count: int = 5):
    """把真实 seekdb 集合适配成与离线基准相同的检索器接口。"""

    def vector(query: str, _analysis: QueryAnalysis):
        return _evidence_from_results(
            collection.query(
                query_texts=[query],
                n_results=candidate_count,
            ),
            "seekdb-vector",
        )

    def hybrid(query: str, analysis: QueryAnalysis):
        keyword = _seekdb_keyword(query, analysis)
        metadata_filter = _seekdb_filter(analysis)
        query_branch: dict[str, object] = {
            "where_document": {"$contains": keyword},
            "n_results": candidate_count,
        }
        vector_branch: dict[str, object] = {
            "query_texts": [query],
            "n_results": candidate_count,
        }
        if metadata_filter:
            query_branch["where"] = metadata_filter
            vector_branch["where"] = metadata_filter
        return _evidence_from_results(
            collection.hybrid_search(
                query=query_branch,
                knn=vector_branch,
                rank={"rrf": {}},
                n_results=candidate_count,
            ),
            "seekdb-hybrid",
        )

    return vector, hybrid


def run_offline_benchmark(config: BenchmarkConfig):
    return run_retrieval_triangle_benchmark(
        load_evaluation_cases(CASES_PATH),
        vector_retriever=vector_retrieve,
        hybrid_retriever=hybrid_retrieve,
        config=config,
        backend="offline-simulator",
    )


def run_seekdb_benchmark(config: BenchmarkConfig):
    """读取已初始化的 D3 集合并实测数据库检索，不执行任何写入。"""
    from d3_1_ingest import COLLECTION_NAME, create_db_client

    with create_db_client() as database:
        if not database.has_collection(COLLECTION_NAME):
            raise RuntimeError("未找到 D3 知识库，请先运行 d3_1_ingest.py")
        collection = database.get_collection(COLLECTION_NAME)
        vector, hybrid = build_seekdb_retrievers(
            collection,
            candidate_count=max(5, config.top_k),
        )
        return run_retrieval_triangle_benchmark(
            load_evaluation_cases(CASES_PATH),
            vector_retriever=vector,
            hybrid_retriever=hybrid,
            config=config,
            backend="seekdb",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        warmup_rounds=args.warmup,
        measurement_rounds=args.rounds,
        top_k=args.top_k,
        cost=CostAssumptions(
            input_price_per_million_tokens=args.input_price_per_million,
            currency=args.currency,
        ),
    )
    report = (
        run_seekdb_benchmark(config)
        if args.backend == "seekdb"
        else run_offline_benchmark(config)
    )
    output = args.output or (
        SEEKDB_REPORT_PATH if args.backend == "seekdb" else REPORT_PATH
    )
    write_markdown_report(report, output)
    by_name = {result.strategy: result for result in report.strategies}
    vector = by_name["vector"]
    hybrid = by_name["hybrid"]
    print(f">>> 已完成 {report.case_count} 条可回答案例的检索三角实验")
    print(
        f">>> Hit@1：纯向量 {vector.hit_at_1:.2f} / "
        f"混合检索 {hybrid.hit_at_1:.2f}"
    )
    print(
        f">>> P95：纯向量 {vector.latency_p95_ms:.4f} ms / "
        f"混合检索 {hybrid.latency_p95_ms:.4f} ms"
    )
    print(f">>> 报告：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
