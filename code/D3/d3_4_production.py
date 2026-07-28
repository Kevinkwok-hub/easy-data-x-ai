import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
import pyseekdb
import json
from openai import OpenAI

# ============================================================
# d3_4：从实验到生产——几个关键工程要点
#
# 演示：
#   1. 工具描述（Tool Description）质量对 Agent 行为的影响
#   2. top_k 参数的取舍
#   3. 数据更新：增量写入而非全量重建
#
# 运行前：先运行 d3_1_ingest.py 写入数据
# 运行：python d3_4_production.py
# ============================================================


# ---------- 0. 连接知识库 ----------

DATABASE_PATH = Path(__file__).resolve().parent / "d3_seekdb"
COLLECTION_NAME = "d3_product_kb"
MODEL = "deepseek-ai/DeepSeek-V3"


def create_db_client():
    """创建路径稳定的 seekdb 客户端。"""
    return pyseekdb.Client(path=str(DATABASE_PATH))


def create_model_client():
    return OpenAI(
        api_key=Config.SILICONFLOW_API_KEY,
        base_url=Config.SILICONFLOW_BASE_URL,
    )


# ---------- 1. 工具描述质量的影响 ----------

def ask_with_tool_desc(
    question: str,
    tool_description: str,
    label: str,
    *,
    api_client,
) -> str:
    """用指定的工具描述发起 Agent 调用"""
    tools = [{
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询文本"}
                },
                "required": ["query"]
            }
        }
    }]

    messages = [
        {"role": "system", "content": "你是一个技术助手，回答用户关于产品的问题。"},
        {"role": "user", "content": question}
    ]

    response = api_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    choices = getattr(response, "choices", None)
    if not choices:
        return f"[{label}] ⚠️  模型未返回有效内容"
    message = getattr(choices[0], "message", None)
    if message is None:
        return f"[{label}] ⚠️  模型未返回有效内容"

    if getattr(message, "tool_calls", None):
        tool_call = message.tool_calls[0]
        function = getattr(tool_call, "function", None)
        function_name = getattr(function, "name", "")
        if function_name != "search_knowledge_base":
            return f"[{label}] ⚠️  未知工具：{function_name or '未提供名称'}"
        try:
            arguments = json.loads(getattr(function, "arguments", ""))
        except (TypeError, json.JSONDecodeError):
            return f"[{label}] ⚠️  工具调用参数不是有效 JSON"
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not isinstance(query, str) or not query.strip():
            return f"[{label}] ⚠️  工具调用缺少非空字符串参数 query"
        return f"[{label}] ✅ Agent 调用了工具，查询：\"{query.strip()}\""

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        return f"[{label}] ⚠️  模型未返回有效内容"
    return f"[{label}] ⚠️  Agent 直接回答（未调用工具）：\"{content[:80]}...\""


new_doc = {
    "id": "kb_013",
    "content": "OB-4.3.0 版本新特性：支持向量索引加速，引入自适应压缩算法，存储空间减少 30%。与 OB-4.2.x 完全兼容，支持滚动升级。",
    "doc_type": "release_notes",
    "version": "4.3.0"
}


def upsert_document(collection, document):
    """原子写入单条文档，避免检查后写入竞态。"""
    collection.upsert(
        ids=[document["id"]],
        documents=[document["content"]],
        metadatas=[{
            "doc_type": document["doc_type"],
            "version": document["version"],
        }],
    )


def main():
    with create_db_client() as database:
        if not database.has_collection(COLLECTION_NAME):
            print("❌ 未找到知识库，请先运行 d3_1_ingest.py 写入数据")
            return
        collection = database.get_collection(COLLECTION_NAME)
        client = create_model_client()
        print(
            f">>> 已连接知识库：{COLLECTION_NAME}，"
            f"共 {collection.count()} 条文档\n"
        )

        print("=" * 60)
        print("【要点一】工具描述质量对 Agent 行为的影响")
        print("=" * 60)
        test_q = "OB-4.2.1 版本和旧版本兼容吗？"
        print(
            ask_with_tool_desc(
                test_q,
                "查询数据库",
                "模糊描述",
                api_client=client,
            )
        )
        clear_desc = (
            "从产品知识库中检索相关信息。"
            "当用户询问产品功能、错误码、版本信息、性能优化、营收数据等问题时使用。"
        )
        print(
            ask_with_tool_desc(
                test_q,
                clear_desc,
                "清晰描述",
                api_client=client,
            )
        )

        print("\n【要点二】top_k 参数的取舍")
        query = "数据库性能优化"
        for top_k in [1, 3, 5, 8]:
            results = collection.hybrid_search(
                query={
                    "where_document": {"$contains": "性能"},
                    "n_results": top_k + 2,
                },
                knn={"query_texts": [query], "n_results": top_k + 2},
                rank={"rrf": {}},
                n_results=top_k,
            )
            docs = results.get("documents", [[]])[0]
            print(f"  top_k={top_k}：返回 {len(docs)} 条")

        print("\n【要点三】增量更新知识库")
        upsert_document(collection, new_doc)
        print(f"增量写入或更新 1 条文档后：{collection.count()} 条")

    print("\n✅ d3_4 完成！三个生产要点演示结束。")


if __name__ == "__main__":
    main()
