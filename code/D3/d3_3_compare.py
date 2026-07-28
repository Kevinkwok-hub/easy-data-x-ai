import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seekdb_runtime import create_seekdb_client

# ============================================================
# d3_3：对比实验——纯向量检索 vs 混合检索（含元数据过滤）
#
# 演示：
#   1. 同一组查询，分别用纯向量检索和混合/增强检索
#   2. 直观看到两种策略在精确匹配场景下的结果差距
#   3. 展示三种检索能力的适用场景：
#      - 向量语义：理解"意思"
#      - 全文关键词：精确匹配错误码、函数名等
#      - 元数据过滤：精确匹配版本号、分类等结构化字段
#
# 运行前：先运行 d3_1_ingest.py 写入数据
# 运行：python3 d3_3_compare.py
# ============================================================


# ---------- 0. 连接知识库 ----------

DATABASE_PATH = Path(__file__).resolve().parent / "d3_seekdb"
COLLECTION_NAME = "d3_product_kb"
collection = None


def create_db_client():
    """创建路径稳定的 seekdb 客户端。"""
    return create_seekdb_client(path=DATABASE_PATH)


# ---------- 1. 辅助函数 ----------

def get_top1_snippet(results: dict) -> str:
    """提取 Top-1 结果的前 65 个字符"""
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "（无结果）"
    return docs[0][:65] + "..."


def _require_collection(target_collection=None):
    active_collection = target_collection if target_collection is not None else collection
    if active_collection is None:
        raise RuntimeError("请先初始化知识库集合")
    return active_collection


def vector_only(query: str, n_results: int = 3, *, target_collection=None) -> dict:
    """纯向量检索：只用语义相似度"""
    return _require_collection(target_collection).query(
        query_texts=[query],
        n_results=n_results,
    )


def hybrid_with_keyword(
    query: str,
    keyword: str,
    n_results: int = 3,
    *,
    target_collection=None,
) -> dict:
    """混合检索：向量语义 + 全文关键词，RRF 融合"""
    return _require_collection(target_collection).hybrid_search(
        query={"where_document": {"$contains": keyword}, "n_results": 5},
        knn={"query_texts": [query], "n_results": 5},
        rank={"rrf": {}},
        n_results=n_results,
    )


def vector_with_metadata(
    query: str,
    metadata_filter: dict,
    n_results: int = 3,
    *,
    target_collection=None,
) -> dict:
    """向量检索 + 元数据过滤：语义搜索 + 结构化字段精确匹配"""
    return _require_collection(target_collection).query(
        query_texts=[query],
        where=metadata_filter,
        n_results=n_results,
    )


# ---------- 2. 对比实验 ----------

test_cases = [
    {
        "desc": "场景一：精确错误码查询",
        "query": "E-4012 错误怎么解决",
        "vector_fn": lambda q: vector_only(q),
        "enhanced_fn": lambda q: hybrid_with_keyword(q, "E-4012"),
        "enhanced_label": "混合检索（全文关键词 E-4012）",
        "correct": "E-4012",
    },
    {
        "desc": "场景二：精确季度数据查询",
        "query": "2024年Q3的营收情况",
        "vector_fn": lambda q: vector_only(q),
        "enhanced_fn": lambda q: hybrid_with_keyword(q, "Q3"),
        "enhanced_label": "混合检索（全文关键词 Q3）",
        "correct": "Q3",
    },
    {
        "desc": "场景三：精确版本号查询（版本号含点号，全文搜索分词器不支持）",
        "query": "OB-4.2.1 版本和旧版本兼容吗",
        "vector_fn": lambda q: vector_only(q),
        "enhanced_fn": lambda q: vector_with_metadata(q, {"version": "4.2.1"}),
        "enhanced_label": "向量检索 + 元数据过滤（version=4.2.1）",
        "correct": "4.2.1",
    },
    {
        "desc": "场景四：精确函数名查询",
        "query": "DBMS_HYBRID_SEARCH 函数的用法",
        "vector_fn": lambda q: vector_only(q),
        "enhanced_fn": lambda q: hybrid_with_keyword(q, "DBMS_HYBRID_SEARCH"),
        "enhanced_label": "混合检索（全文关键词 DBMS_HYBRID_SEARCH）",
        "correct": "DBMS_HYBRID_SEARCH",
    },
    {
        "desc": "场景五：纯语义查询（对照组，两种方式差异不大）",
        "query": "怎么优化数据库的查询性能",
        "vector_fn": lambda q: vector_only(q),
        "enhanced_fn": lambda q: hybrid_with_keyword(q, "性能"),
        "enhanced_label": "混合检索（全文关键词 性能）",
        "correct": "性能",
    },
]


def run_comparison(target_collection):
    """运行检索对比实验。"""
    global collection
    previous_collection = collection
    collection = target_collection
    try:
        vector_hits = 0
        enhanced_hits = 0

        for case in test_cases:
            query = case["query"]
            v_top1 = get_top1_snippet(case["vector_fn"](query))
            e_top1 = get_top1_snippet(case["enhanced_fn"](query))
            v_hit = case["correct"] in v_top1
            e_hit = case["correct"] in e_top1
            vector_hits += int(v_hit)
            enhanced_hits += int(e_hit)

            print(f"\n【{case['desc']}】")
            print(f"  查询：\"{query}\"")
            print(f"  纯向量检索 Top-1 {'✅' if v_hit else '❌'}：{v_top1}")
            print(
                f"  {case['enhanced_label']} Top-1 "
                f"{'✅' if e_hit else '❌'}：{e_top1}"
            )
        return vector_hits, enhanced_hits
    finally:
        collection = previous_collection


def main():
    with create_db_client() as database:
        if not database.has_collection(COLLECTION_NAME):
            print("❌ 未找到知识库，请先运行 d3_1_ingest.py 写入数据")
            return 1
        target_collection = database.get_collection(COLLECTION_NAME)
        print(
            f">>> 已连接知识库：{COLLECTION_NAME}，"
            f"共 {target_collection.count()} 条文档\n"
        )
        print("=" * 70)
        print("对比实验：纯向量检索 vs 增强检索（混合/元数据过滤）")
        print("=" * 70)
        vector_hits, enhanced_hits = run_comparison(target_collection)

    print()
    print("=" * 70)
    print("汇总结果：")
    print(f"  纯向量检索命中率：{vector_hits}/{len(test_cases)}")
    print(f"  增强检索命中率：  {enhanced_hits}/{len(test_cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
