import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seekdb_runtime import create_seekdb_client, require_destructive_seekdb_access
from rag_data import knowledge_chunks

# ============================================================
# d3_1：构建知识库
#
# 演示：
#   1. 将预处理好的文档片段写入 seekdb
#   2. 知识库包含版本说明、错误码、财务数据、最佳实践等内容
#   3. 为后续 d3_2（Agentic RAG）和 d3_3（对比实验）提供数据基础
#
# 运行：python d3_1_ingest.py
# ============================================================


# ---------- 1. 准备示例数据集 ----------
# 模拟一个技术产品的知识库，包含多种类型的文档片段
# 这些内容已经完成了分块处理，每条记录就是一个 chunk

# ---------- 2. 初始化 seekdb 并写入数据 ----------

DATABASE_PATH = Path(__file__).resolve().parent / "d3_seekdb"
COLLECTION_NAME = "d3_product_kb"


def create_db_client():
    """创建路径稳定的 seekdb 客户端。"""
    return create_seekdb_client(path=DATABASE_PATH)


def build_knowledge_base(database, *, embedding_function=None):
    """复用固定集合，并用 upsert 幂等写入知识片段。"""
    collection_kwargs = {"name": COLLECTION_NAME}
    if embedding_function is not None:
        collection_kwargs["embedding_function"] = embedding_function
    collection = database.get_or_create_collection(**collection_kwargs)
    collection.upsert(
        ids=[chunk["id"] for chunk in knowledge_chunks],
        documents=[chunk["content"] for chunk in knowledge_chunks],
        metadatas=[
            {"doc_type": chunk["doc_type"], "version": chunk["version"]}
            for chunk in knowledge_chunks
        ],
    )
    return collection


def main():
    require_destructive_seekdb_access("写入 D3 产品知识库")
    print(">>> 正在初始化 seekdb...")
    with create_db_client() as database:
        collection = build_knowledge_base(database)
        print(f">>> 集合已就绪：{COLLECTION_NAME}\n")
        print(f">>> 已写入或更新 {collection.count()} 个知识片段")
        print()

        type_counts = Counter(chunk["doc_type"] for chunk in knowledge_chunks)
        print(">>> 知识库内容分布：")
        for doc_type, count in type_counts.items():
            print(f"    {doc_type}: {count} 条")

    print()
    print(">>> d3_1 完成！知识库已就绪，可运行 d3_2 / d3_3 继续体验。")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
