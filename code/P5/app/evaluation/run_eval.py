"""CLI entry point for running the offline Knowledge Agent evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.agent import DEFAULT_KNOWLEDGE_BASE, KnowledgeAgent
from app.evaluation.evaluators import evaluate_cases, load_eval_cases
from app.evaluation.report import write_evaluation_reports


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "eval_dataset.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def run(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    knowledge_base_path: str | Path = DEFAULT_KNOWLEDGE_BASE,
) -> list[Path]:
    """执行一次完整离线评测，并返回生成的报告路径。

    这个函数是 CLI 和测试共用的薄封装：先加载 JSONL 样本，再创建 Mock Agent，
    最后把聚合指标写成 JSON/Markdown。保持它无外部服务依赖，课程读者才能稳定复现。
    """
    cases = load_eval_cases(Path(dataset_path))
    agent = KnowledgeAgent(knowledge_base_path)
    report = evaluate_cases(cases, agent)
    return write_evaluation_reports(report, Path(output_dir))


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数并运行评测。

    main 只负责命令行交互，不直接写指标计算逻辑；这样后续 FastAPI 或 CI 想复用评测流程时，
    可以直接调用 `run()`，不用模拟 argparse。
    """
    parser = argparse.ArgumentParser(description="Run the P5 offline Knowledge Agent evaluation.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to eval_dataset.jsonl.")
    parser.add_argument("--knowledge-base", default=str(DEFAULT_KNOWLEDGE_BASE), help="Path to knowledge_base.json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation reports.")
    args = parser.parse_args(argv)

    paths = run(args.dataset, args.output_dir, args.knowledge_base)
    for path in paths:
        print(f"Generated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
