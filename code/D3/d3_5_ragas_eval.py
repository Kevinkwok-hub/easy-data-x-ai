"""D3 可复现的 RAGAS / 检索评测入口。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from d3_eval_core import (
    RAGAS_METRIC_KEYS,
    RETRIEVAL_METRIC_KEYS,
    aggregate_by_scenario,
    aggregate_rows,
    attach_ragas_scores,
    compare_strategies,
    compute_retrieval_metrics,
    extract_lexical_anchor,
    extract_metadata_filter,
    is_finite_number,
    load_json,
    scenario_type,
    sha256_file,
    sha256_text,
    validate_dataset,
    write_json,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATABASE_PATH = Path(__file__).resolve().parent / "d3_seekdb"
EVAL_PATH = DATA_DIR / "eval_dataset.json"
RUNS_DIR = DATA_DIR / "runs"
DEFAULT_COLLECTION = "d3_product_kb"


def connect_collection(collection_name: str):
    # 延迟导入，避免单元测试 import 本模块时提前绑定真实 pyseekdb。
    from seekdb_runtime import create_seekdb_client

    db = create_seekdb_client(path=DATABASE_PATH)
    if not db.has_collection(collection_name):
        raise RuntimeError("未找到知识库，请先运行 d3_1_ingest.py 写入数据")
    return db.get_collection(collection_name)


def docs_and_ids(results: dict[str, Any]) -> tuple[list[str], list[str]]:
    """展平 seekdb 查询返回的二维 documents / ids。"""
    documents = results.get("documents", [[]])
    ids = results.get("ids", [[]])
    flat_docs = documents[0] if documents else []
    flat_ids = ids[0] if ids else []
    return list(flat_docs or []), list(flat_ids or [])


def vector_search(collection, question: str, top_k: int) -> tuple[list[str], list[str]]:
    return docs_and_ids(collection.query(query_texts=[question], n_results=top_k))


def hybrid_search(collection, question: str, top_k: int) -> tuple[list[str], list[str]]:
    """使用短全文查询、可选版本过滤与向量查询执行公平的混合检索。"""
    anchor = extract_lexical_anchor(question)
    if not anchor:
        raise ValueError(f"无法从问题提取独立全文查询: {question}")
    metadata_filter = extract_metadata_filter(question)
    candidate_k = max(5, top_k + 2)
    query = {
        "where_document": {"$contains": anchor},
        "n_results": candidate_k,
    }
    knn = {"query_texts": [question], "n_results": candidate_k}
    if metadata_filter:
        query["where"] = metadata_filter
        knn["where"] = metadata_filter
    return docs_and_ids(
        collection.hybrid_search(
            query=query,
            knn=knn,
            rank={"rrf": {}},
            n_results=top_k,
        )
    )


def build_id_to_doc(collection) -> dict[str, str]:
    """读取集合全貌，供运行工件记录使用。"""
    payload = collection.get()
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    return {doc_id: document for doc_id, document in zip(ids, documents)}


def build_id_to_metadata(collection) -> dict[str, dict[str, Any]]:
    """读取文档元数据，供知识库指纹记录使用。"""
    payload = collection.get()
    ids = payload.get("ids") or []
    metadatas = payload.get("metadatas") or []
    return {doc_id: metadata or {} for doc_id, metadata in zip(ids, metadatas)}


def collection_fingerprint(
    id_to_doc: dict[str, str],
    id_to_metadata: dict[str, dict[str, Any]],
) -> str:
    payload = {
        doc_id: {
            "content_hash": sha256_text(id_to_doc[doc_id]),
            "metadata": id_to_metadata.get(doc_id, {}),
        }
        for doc_id in sorted(id_to_doc)
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_generation_client():
    """构造生成答案的 OpenAI 兼容客户端，与 RAGAS 评测共用配置。"""
    config = Config.get_ragas_evaluator_config()
    if config["missing"]:
        raise RuntimeError("缺少 RAGAS 配置: " + ", ".join(config["missing"]))
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("未安装 openai，请先安装 code/requirements.txt") from error
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    ), config["llm_model"]


def generate_answer(client, model: str, question: str, contexts: list[str]) -> str:
    evidence = "\n".join(f"- {item}" for item in contexts) or "- （无检索结果）"
    prompt = (
        "你是产品技术文档助手。只依据给定证据回答，证据不足时明确说明无法确定，"
        "不要编造错误码、版本号或时间。\n\n证据：\n"
        f"{evidence}\n\n问题：{question}"
    )
    # Qwen3 等思考模型需关闭 thinking，否则长上下文下可能长时间无响应。
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def retrieve_case(
    collection,
    case: dict[str, Any],
    strategy: str,
    top_k: int,
) -> tuple[list[str], list[str]]:
    question = case["question"]
    if strategy == "vector":
        return vector_search(collection, question, top_k)
    if strategy == "hybrid":
        return hybrid_search(collection, question, top_k)
    raise ValueError(f"未知策略: {strategy}")


def build_result_row(
    case: dict[str, Any],
    strategy: str,
    documents: list[str],
    ids: list[str],
    answer: str,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把检索与生成结果压缩成可落盘的逐例行。"""
    return {
        "case_id": case["id"],
        "scenario": scenario_type(case),
        "strategy": strategy,
        "question": case["question"],
        "reference": case["reference"],
        "retrieved_ids": ids,
        "retrieved_contexts": documents,
        "answer": answer,
        "retrieval_metrics": compute_retrieval_metrics(case, ids),
        "ragas_scores": {},
        "failure": failure,
    }


def record_failure(stage: str, detail: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "detail": detail,
        "message": f"{stage} failed; inspect local logs for provider details",
    }


def collect_strategy_rows(
    collection,
    cases: list[dict[str, Any]],
    strategy: str,
    top_k: int,
    client,
    model: str,
    generate: bool,
) -> list[dict[str, Any]]:
    """收集单策略逐例结果，保留原始检索 ID 与确定性指标。"""
    rows = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        print(f"  [{strategy}] {index}/{total} {case['id']}", flush=True)
        try:
            documents, ids = retrieve_case(collection, case, strategy, top_k)
            answer = ""
            failure = None
            if generate:
                try:
                    answer = generate_answer(client, model, case["question"], documents)
                except Exception as error:  # noqa: BLE001 - 记录后继续
                    failure = record_failure("generation", type(error).__name__)
            rows.append(build_result_row(case, strategy, documents, ids, answer, failure))
        except Exception as error:  # noqa: BLE001 - 记录后继续
            rows.append(
                build_result_row(
                    case,
                    strategy,
                    [],
                    [],
                    "",
                    record_failure("retrieval", type(error).__name__),
                )
            )
    return rows


def ensure_no_pipeline_failures(rows_by_strategy: dict[str, list[dict[str, Any]]]) -> None:
    """任何检索或生成失败都阻止发布不完整 benchmark。"""
    failures = []
    for strategy, rows in rows_by_strategy.items():
        for row in rows:
            failure = row.get("failure")
            if failure:
                failures.append(
                    f"{strategy}/{row.get('case_id')}: "
                    f"{failure.get('stage', 'unknown')} "
                    f"({failure.get('detail', 'unknown')})"
                )
    if failures:
        raise RuntimeError("评测流水线存在失败:\n" + "\n".join(failures))


def ensure_complete_metric_scores(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...] = RAGAS_METRIC_KEYS,
) -> None:
    """要求每条样本的每个 RAGAS 指标都有有限分数。"""
    missing = []
    for row in rows:
        for key in keys:
            value = row.get("ragas_scores", {}).get(key)
            if not is_finite_number(value):
                missing.append(f"{row.get('case_id')}/{key}")
    if missing:
        raise RuntimeError("RAGAS 指标未完整评分: " + ", ".join(missing))


def build_ragas_dependencies():
    """显式构造评测 LLM 与 embedding，避免使用 RAGAS 默认 OpenAI 值。"""
    config = Config.get_ragas_evaluator_config()
    if config["missing"]:
        raise RuntimeError("缺少 RAGAS 配置: " + ", ".join(config["missing"]))
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as error:
        raise RuntimeError("RAGAS 模式需要 ragas 与 langchain-openai") from error
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["llm_model"],
            temperature=0,
            max_tokens=512,
            extra_body={"enable_thinking": False},
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["embedding_model"],
        )
    )
    return llm, embeddings


def run_ragas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """运行 RAGAS，并把逐例分数写回原始行。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = build_ragas_dependencies()
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["retrieved_contexts"],
                "ground_truth": row["reference"],
            }
            for row in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )
    scores = getattr(result, "scores", None)
    if not scores:
        raise RuntimeError("no metric scores returned")
    attach_ragas_scores(rows, list(scores))
    ensure_complete_metric_scores(rows)
    return rows


def build_strategy_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """生成整体、分场景指标，且只在 ragas 模式汇总真实 RAGAS 分数。"""
    keys = list(RETRIEVAL_METRIC_KEYS)
    if mode == "ragas":
        keys.extend(RAGAS_METRIC_KEYS)
    return {
        "overall": aggregate_rows(rows, keys),
        "by_scenario": aggregate_by_scenario(rows, keys),
    }


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    """按阶段统计运行中的失败次数，供 summary 侧栏展示。"""
    counts: dict[str, int] = {}
    for row in rows:
        failure = row.get("failure")
        if not failure:
            continue
        stage = str(failure.get("stage", "unknown"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def build_summary(
    dataset: dict[str, Any],
    mode: str,
    top_k: int,
    vector_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """生成可落盘且可由逐例结果复算的汇总对象。"""
    keys = list(RETRIEVAL_METRIC_KEYS)
    if mode == "ragas":
        keys.extend(RAGAS_METRIC_KEYS)
    return {
        "schema_version": "2.0",
        "evaluation_mode": mode,
        "dataset_id": dataset["dataset_id"],
        "n_cases": len(dataset["cases"]),
        "top_k": top_k,
        "strategies": {
            "vector": build_strategy_summary(vector_rows, mode),
            "hybrid": build_strategy_summary(hybrid_rows, mode),
        },
        "paired_comparison": compare_strategies(vector_rows, hybrid_rows, keys),
        "failures": {
            "vector": summarize_failures(vector_rows),
            "hybrid": summarize_failures(hybrid_rows),
        },
    }


def create_run_directory(output_dir: Path | None) -> Path:
    run_dir = output_dir or (RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_case_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def installed_dependency_versions() -> dict[str, str | None]:
    """记录关键依赖的版本；缺失包以 null 表示。"""
    versions: dict[str, str | None] = {}
    for package in ("ragas", "langchain-openai", "openai", "pyseekdb"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def write_run_artifacts(
    run_dir: Path,
    dataset_path: Path,
    collection_hash: str,
    collection_count: int,
    summary: dict[str, Any],
    vector_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    generator_model: str | None,
) -> dict[str, Any]:
    """写入不含密钥的 manifest、逐例结果和可读摘要。"""
    evaluator = Config.get_ragas_evaluator_config()
    manifest = {
        "schema_version": "2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "id": summary["dataset_id"],
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "knowledge_base": {
            "collection": args.collection,
            "document_count": collection_count,
            "fingerprint": collection_hash,
        },
        "mode": args.mode,
        "strategy_parameters": {
            "top_k": summary["top_k"],
            "vector": {"query": "original_question"},
            "hybrid": {
                "lexical_anchor": "extract_lexical_anchor(original_question)",
                "rank_fusion": "rrf",
            },
        },
        "generator_model": generator_model if args.mode == "ragas" else None,
        "evaluator_model": evaluator["llm_model"] if args.mode == "ragas" else None,
        "evaluator_embedding_model": (
            evaluator["embedding_model"] if args.mode == "ragas" else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "dependencies": installed_dependency_versions(),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    write_case_results(run_dir / "case_results.vector.jsonl", vector_rows)
    write_case_results(run_dir / "case_results.hybrid.jsonl", hybrid_rows)
    (run_dir / "summary.md").write_text(
        format_summary_markdown(summary),
        encoding="utf-8",
    )
    publish_path = getattr(args, "publish_summary", None)
    if publish_path is not None:
        write_json(
            publish_path,
            {
                "schema_version": "2.0",
                "source": "generated by d3_5_ragas_eval.py from complete run artifacts",
                "created_at": manifest["created_at"],
                "dataset": {
                    "id": manifest["dataset"]["id"],
                    "sha256": manifest["dataset"]["sha256"],
                    "n_cases": summary["n_cases"],
                    "top_k": summary["top_k"],
                },
                "knowledge_base": manifest["knowledge_base"],
                "models": {
                    "generator_model": manifest["generator_model"],
                    "evaluator_model": manifest["evaluator_model"],
                    "evaluator_embedding_model": manifest[
                        "evaluator_embedding_model"
                    ],
                },
                "runtime": manifest["runtime"],
                "strategies": summary["strategies"],
                "paired_comparison": summary["paired_comparison"],
                "failures": summary["failures"],
                "case_results": {
                    "vector": vector_rows,
                    "hybrid": hybrid_rows,
                },
            },
        )
    return manifest


def format_summary_markdown(summary: dict[str, Any]) -> str:
    """生成适合课程稿粘贴的简短 Markdown 摘要。"""
    lines = [
        f"# D3 evaluation ({summary['evaluation_mode']})",
        "",
        f"- dataset: `{summary['dataset_id']}`",
        f"- n_cases: {summary['n_cases']}",
        f"- top_k: {summary['top_k']}",
        "",
        "## overall metrics",
    ]
    for strategy in ("vector", "hybrid"):
        metrics = summary["strategies"][strategy]["overall"]["metrics"]
        lines.append(f"### {strategy}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## paired comparison (hybrid - vector)",
            "",
            "| metric | n | mean delta | 95% CI | p-value | significant |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for key, comparison in summary["paired_comparison"].items():
        ci = comparison["ci95"]
        ci_text = "-" if ci is None else f"[{ci[0]}, {ci[1]}]"
        lines.append(
            f"| {key} | {comparison['n_pairs']} | {comparison['mean_delta']} | "
            f"{ci_text} | {comparison['p_value']} | "
            f"{comparison['significant_at_0_05']} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(summary: dict[str, Any]) -> None:
    """在终端显示整体确定性检索指标与 RAGAS 指标。"""
    print("\n" + "=" * 72)
    print(f"D3 evaluation mode={summary['evaluation_mode']} dataset={summary['dataset_id']}")
    print("=" * 72)
    for strategy in ("vector", "hybrid"):
        metrics = summary["strategies"][strategy]["overall"]["metrics"]
        print(f"\n[{strategy}]")
        for key, value in metrics.items():
            print(f"  {key:<32} {value}")
    failures = summary.get("failures", {})
    print("\n  failures                         ", failures)


def check_config() -> int:
    """只检查 RAGAS 配置完整性，不发起网络请求。"""
    config = Config.get_ragas_evaluator_config()
    print("RAGAS evaluator config ready" if not config["missing"] else "missing:")
    for item in config["missing"]:
        print(f"- {item}")
    if not config["missing"]:
        print(f"llm_model: {config['llm_model']}")
        print(f"embedding_model: {config['embedding_model']}")
        print(f"base_url: {config['base_url']}")
    print("RAGAS 检查不输出密钥或落盘；未触发任何模型调用。")
    return 0 if not config["missing"] else 1


def run_evaluation(args: argparse.Namespace) -> int:
    """执行严格 RAGAS 或仅检索指标的真实评测。"""
    if args.publish_summary is not None and args.mode != "ragas":
        raise ValueError("--publish-summary 仅支持 --mode ragas")

    dataset = load_json(args.dataset)
    top_k = args.top_k or int(dataset.get("default_top_k", 4))

    collection = connect_collection(args.collection)
    id_to_doc = build_id_to_doc(collection)
    id_to_metadata = build_id_to_metadata(collection)
    fingerprint = collection_fingerprint(id_to_doc, id_to_metadata)

    errors = validate_dataset(dataset, known_ids=set(id_to_doc), top_k=top_k)
    if errors:
        raise ValueError("数据集校验失败:\n" + "\n".join(errors))

    generate = args.mode == "ragas"
    client = model = None
    if generate:
        client, model = build_generation_client()

    vector_rows = collect_strategy_rows(
        collection,
        dataset["cases"],
        "vector",
        top_k,
        client,
        model or "",
        generate,
    )
    hybrid_rows = collect_strategy_rows(
        collection,
        dataset["cases"],
        "hybrid",
        top_k,
        client,
        model or "",
        generate,
    )
    ensure_no_pipeline_failures({"vector": vector_rows, "hybrid": hybrid_rows})
    if args.mode == "ragas":
        try:
            print(">>> running RAGAS on vector rows...", flush=True)
            run_ragas(vector_rows)
            print(">>> running RAGAS on hybrid rows...", flush=True)
            run_ragas(hybrid_rows)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"RAGAS 评测运行失败: {error}") from error

    summary = build_summary(
        dataset,
        args.mode,
        top_k,
        vector_rows,
        hybrid_rows,
    )
    run_dir = create_run_directory(args.output_dir)
    write_run_artifacts(
        run_dir,
        args.dataset,
        fingerprint,
        len(id_to_doc),
        summary,
        vector_rows,
        hybrid_rows,
        args,
        model,
    )
    print_summary(summary)
    print(f"\nartifacts: {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    """解析评测脚本的运行参数。"""
    parser = argparse.ArgumentParser(description="D3 可复现 RAGAS / 检索评测")
    parser.add_argument(
        "--mode",
        choices=("ragas", "retrieval"),
        default="ragas",
        help="ragas 为完整评测；retrieval 只计算金标检索指标",
    )
    parser.add_argument("--check-config", action="store_true", help="只检查 RAGAS 配置")
    parser.add_argument("--dataset", type=Path, default=EVAL_PATH, help="评测数据集路径")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="seekdb 集合名称")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖数据集默认值")
    parser.add_argument("--output-dir", type=Path, default=None, help="运行工件输出目录")
    parser.add_argument(
        "--publish-summary",
        type=Path,
        default=None,
        help="完整成功后额外写入可提交的溯源摘要",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_config:
        return check_config()
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
