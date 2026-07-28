import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seekdb_runtime import create_seekdb_client, require_destructive_seekdb_access

if __package__:
    from .db_lifecycle import close_database as _close_database
else:
    from db_lifecycle import close_database as _close_database


D2_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D2_DIR / "seekdb"
COLLECTION_NAME = "d2_knowledge_base"
RAW_DOCS = [
    ("错误码 E-4012 表示数据库连接超时。解决方案：检查网络和端口，超时建议设为 30 秒。", "error_codes", "4.2"),
    ("错误码 E-4013 表示认证失败。请检查用户名、密码和账户锁定状态。", "error_codes", "4.2"),
    ("错误码 E-4011 表示连接被拒绝。请确认数据库服务和防火墙规则。", "error_codes", "4.2"),
    ("数据库查询性能优化指南：为高频 WHERE 条件列建立索引，避免在索引列上使用函数。", "best_practices", "4.2"),
    ("访问控制架构设计：基于 RBAC 实现用户权限管理，并按最小权限原则分配角色。", "architecture", "4.1"),
    ("OB-4.2.1 支持在线 DDL，并改进并行查询引擎。升级前请备份数据。", "release_notes", "4.2.1"),
    ("数据备份与恢复：每天全量备份，每小时增量备份，并使用独立存储介质。", "best_practices", "4.2"),
    ("连接池配置：最大连接数建议为 CPU 核心数的 2-4 倍，连接超时建议 30 秒。", "best_practices", "4.2"),
]


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """将长文档切成带重叠的片段。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk]


def run_demo(db) -> int:
    if db.has_collection(COLLECTION_NAME):
        db.delete_collection(COLLECTION_NAME)
        print(f">>> 已删除旧的集合：{COLLECTION_NAME}")
    collection = db.create_collection(name=COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for doc_index, (text, category, version) in enumerate(RAW_DOCS):
        for chunk_index, chunk in enumerate(chunk_document(text)):
            ids.append(f"doc_{doc_index}_chunk_{chunk_index}")
            texts.append(chunk)
            metadatas.append(
                {
                    "category": category,
                    "version": version,
                    "source_doc_idx": doc_index,
                }
            )
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    stored_count = collection.count()
    print(f">>> 原始文档数：{len(RAW_DOCS)}")
    print(f">>> 切分后片段数：{len(ids)}")
    print(f">>> seekdb 中实际存储条数：{stored_count}")
    if stored_count == len(ids):
        print("✅ d2_1 写入完成，可继续运行 d2_2 / d2_3 / d2_4。")
    else:
        print("⚠️  实际存储条数与预期不一致，请检查写入日志。")
    return stored_count


def main(client_factory=create_seekdb_client, db_path=None) -> int:
    # 路径边界：默认锚定 D2 章节目录，不使用当前工作目录。
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    require_destructive_seekdb_access("重建 D2 演示知识库")
    db = client_factory(path=str(resolved_path))
    try:
        run_demo(db)
        return 0
    finally:
        _close_database(db)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(db_path=_parse_args().db_path))
