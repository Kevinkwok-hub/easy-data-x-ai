import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seekdb_runtime import create_seekdb_client

if __package__:
    from .db_lifecycle import close_database as _close_database
else:
    from db_lifecycle import close_database as _close_database


D2_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D2_DIR / "seekdb"
COLLECTION_NAME = "d2_knowledge_base"


def _documents(results):
    return results.get("documents", [[]])[0]


def run_comparison(
    collection,
    query_text: str,
    keyword: str,
    n_results: int = 3,
):
    """使用自然语言做向量检索，使用短关键词做真实全文条件。"""
    print(f"\n{'#' * 60}\n  查询：「{query_text}」\n{'#' * 60}\n")
    vector_results = collection.query(
        query_texts=query_text,
        n_results=n_results,
    )
    hybrid_results = collection.hybrid_search(
        query={
            "where_document": {"$contains": keyword},
            "n_results": n_results + 2,
        },
        knn={
            "query_texts": [query_text],
            "n_results": n_results + 2,
        },
        rank={"rrf": {}},
        n_results=n_results,
    )
    vector_docs = _documents(vector_results)
    hybrid_docs = _documents(hybrid_results)
    vector_scores = vector_results.get("distances", [[]])[0]
    hybrid_scores = hybrid_results.get("distances", [[]])[0]
    print(f"  {'纯向量搜索':<35}  混合搜索（向量 + 全文）")
    for index in range(max(len(vector_docs), len(hybrid_docs))):
        if index < len(vector_docs):
            vector = f"[{index + 1}] {vector_docs[index][:30]} ({vector_scores[index]:.3f})"
        else:
            vector = ""
        if index < len(hybrid_docs):
            hybrid = f"[{index + 1}] {hybrid_docs[index][:30]} ({hybrid_scores[index]:.3f})"
        else:
            hybrid = ""
        print(f"  {vector:<45}  {hybrid}")
    return vector_results, hybrid_results


def run_demo(collection):
    comparisons = [
        ("错误码 E-4012 的解决方案", "E-4012", "E-4012"),
        ("怎么设计用户权限", "权限", "访问控制"),
        ("数据库性能优化", "性能优化", "性能优化"),
    ]
    rows = []
    for query_text, keyword, expected in comparisons:
        vector, hybrid = run_comparison(collection, query_text, keyword)
        hybrid_docs = _documents(hybrid)
        if hybrid_docs and expected in " ".join(hybrid_docs):
            print(f"✅ 混合搜索结果包含「{expected}」相关内容。\n")
        else:
            print(f"⚠️  本次混合搜索未返回「{expected}」相关内容。\n")
        rows.append((vector, hybrid))
    return rows


def main(client_factory=create_seekdb_client, db_path=None) -> int:
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    db = client_factory(path=str(resolved_path))
    try:
        if not db.has_collection(COLLECTION_NAME):
            print("❌ 未找到知识库，请先运行 d2_1_ingest.py 写入数据")
            return 1
        collection = db.get_collection(COLLECTION_NAME)
        print(f">>> 已连接知识库：{COLLECTION_NAME}，共 {collection.count()} 条文档\n")
        run_demo(collection)
        return 0
    finally:
        _close_database(db)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(db_path=_parse_args().db_path))
