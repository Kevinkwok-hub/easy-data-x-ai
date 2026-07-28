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


def print_results(results: dict, title: str):
    print(f"{'=' * 50}\n  {title}\n{'=' * 50}")
    docs = _documents(results)
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not docs:
        print("  （无结果）")
    for index, (doc, distance, metadata) in enumerate(
        zip(docs, distances, metadatas), 1
    ):
        print(f"  [{index}] 相关度分数：{distance:.4f}")
        print(f"       内容：{doc[:80]}")
        print(
            f"       分类：{metadata.get('category', '-')}  "
            f"版本：{metadata.get('version', '-')}"
        )


def run_demo(collection):
    semantic = collection.query(query_texts="怎么设计用户权限", n_results=3)
    print_results(semantic, "向量搜索：语义理解")
    semantic_docs = _documents(semantic)
    if any("访问控制" in doc or "RBAC" in doc for doc in semantic_docs):
        print("✅ 结果中找到了访问控制/RBAC 相关文档。\n")
    else:
        print("⚠️  本次结果未命中访问控制/RBAC，不能断言语义检索成功。\n")

    error_code = collection.query(
        query_texts="错误码 E-4012 的解决方案",
        n_results=3,
    )
    print_results(error_code, "向量搜索：精确编号")
    docs = _documents(error_code)
    if docs and "E-4012" in docs[0]:
        print("✅ 本次向量搜索第一名恰好是 E-4012；实际排名仍可能受模型影响。\n")
    elif docs:
        print("⚠️  第一名不是 E-4012，展示了纯向量检索处理精确编号的风险。\n")
    else:
        print("⚠️  本次没有返回结果。\n")
    return semantic, error_code


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
