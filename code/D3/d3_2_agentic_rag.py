import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from seekdb_runtime import create_seekdb_client
from openai import OpenAI

# ============================================================
# d3_2：Agentic RAG 完整链路
#
# 演示：
#   1. 定义 search_knowledge_base 工具（Tool Use）
#   2. Agent 自主决定是否调用工具检索知识库
#   3. 基于检索结果生成准确回答
#   4. 对比"需要检索"和"不需要检索"两种场景
#
# 运行前：先运行 d3_1_ingest.py 写入数据
# 运行：python d3_2_agentic_rag.py
# ============================================================


# ---------- 0. 运行配置 ----------

DATABASE_PATH = Path(__file__).resolve().parent / "d3_seekdb"
COLLECTION_NAME = "d3_product_kb"
MODEL = "deepseek-ai/DeepSeek-V3"


# ---------- 2. 定义工具 ----------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从产品知识库中检索相关信息。"
                "当用户询问产品功能、错误码、版本信息、性能优化、营收数据等问题时使用。"
                "查询文本应尽量保留用户问题中的关键词和专有名词（如版本号、错误码）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索知识库的查询文本，应保留用户问题中的关键词和专有名词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ---------- 3. 工具执行函数 ----------

def create_db_client():
    """创建路径稳定的 seekdb 客户端。"""
    return create_seekdb_client(path=DATABASE_PATH)


def create_model_client():
    """创建 OpenAI 兼容模型客户端。"""
    return OpenAI(
        api_key=Config.SILICONFLOW_API_KEY,
        base_url=Config.SILICONFLOW_BASE_URL,
    )


def extract_search_keyword(query: str) -> str:
    """优先提取错误码、季度、版本号和函数名等精确标识符。"""
    error_code = re.search(r"\bE-\d+\b", query, flags=re.IGNORECASE)
    if error_code:
        return error_code.group(0).upper()

    quarter = re.search(r"(?:20\d{2}年?)?(Q[1-4])", query, flags=re.IGNORECASE)
    if quarter:
        return quarter.group(1).upper()

    version = re.search(r"\bOB-\d+(?:\.\d+)+\b", query, flags=re.IGNORECASE)
    if version:
        return version.group(0)

    function_name = re.search(r"\b[A-Z][A-Z0-9_]{2,}\b", query)
    if function_name:
        return function_name.group(0)

    return query.strip()


def execute_search(query: str, target_collection=None) -> str:
    """调用 seekdb 混合检索，返回格式化的检索结果"""
    if target_collection is None:
        with create_db_client() as database:
            if not database.has_collection(COLLECTION_NAME):
                return "知识库尚未初始化，请先运行 d3_1_ingest.py。"
            collection = database.get_collection(COLLECTION_NAME)
            return execute_search(query, collection)

    keyword = extract_search_keyword(query)
    # 使用混合检索：向量语义 + 全文关键词
    results = target_collection.hybrid_search(
        query={
            "where_document": {"$contains": keyword},
            "n_results": 5,
        },
        knn={"query_texts": [query], "n_results": 5},
        rank={"rrf": {}},
        n_results=3,
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return "知识库中未找到相关内容。"

    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(f"[结果 {i+1}]\n{doc}")
    return "\n\n".join(formatted)


def _tool_result(tool_call, search_fn) -> str:
    """安全解析并执行单个工具调用。"""
    function = getattr(tool_call, "function", None)
    function_name = getattr(function, "name", "")
    if function_name != "search_knowledge_base":
        return f"未知工具：{function_name or '未提供名称'}"

    raw_arguments = getattr(function, "arguments", "")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return "工具调用参数不是有效 JSON。"

    if not isinstance(arguments, dict):
        return "工具调用参数必须是 JSON 对象。"

    query_text = arguments.get("query")
    if not isinstance(query_text, str) or not query_text.strip():
        return "工具调用缺少非空字符串参数 query。"

    return search_fn(query_text.strip())


def run_agent_loop(
    question: str,
    *,
    api_client,
    search_fn,
    max_tool_rounds: int = 5,
) -> str:
    """执行支持多工具、多轮和安全失败的 Agent 工具循环。"""
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds 不能小于 0")

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个产品技术文档助手。仅当用户询问产品知识（如产品功能、错误码、"
                "版本信息、性能优化或产品营收数据）时查询知识库；闲聊和非产品问题不要检索。"
                "需要检索时先获取准确信息，再基于结果回答。知识库没有相关信息时请诚实告知。"
                "回答时请引用具体的数据和版本号，不要猜测。"
            )
        },
        {"role": "user", "content": question}
    ]

    tool_rounds = 0
    while True:
        response = api_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return "模型未返回有效内容。"

        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            return "模型未返回有效内容。"

        if tool_rounds >= max_tool_rounds:
            return f"达到最大工具调用轮数（{max_tool_rounds}），任务已停止。"

        call_ids = [getattr(tool_call, "id", None) for tool_call in tool_calls]
        if any(
            not isinstance(call_id, str) or not call_id.strip()
            for call_id in call_ids
        ):
            raise ValueError("模型工具调用缺少有效 id")
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("同一轮工具调用 id 重复")

        messages.append(message)
        for tool_call, call_id in zip(tool_calls, call_ids):
            result = _tool_result(tool_call, search_fn)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
        tool_rounds += 1


def ask_agent(
    question: str,
    *,
    api_client=None,
    search_fn=None,
    max_tool_rounds: int = 5,
) -> str:
    """使用默认依赖执行 Agentic RAG，也允许测试注入离线依赖。"""
    if api_client is None:
        with create_model_client() as owned_client:
            return ask_agent(
                question,
                api_client=owned_client,
                search_fn=search_fn,
                max_tool_rounds=max_tool_rounds,
            )
    model_client = api_client
    if search_fn is not None:
        return run_agent_loop(
            question,
            api_client=model_client,
            search_fn=search_fn,
            max_tool_rounds=max_tool_rounds,
        )

    with create_db_client() as database:
        if not database.has_collection(COLLECTION_NAME):
            return "知识库尚未初始化，请先运行 d3_1_ingest.py。"
        collection = database.get_collection(COLLECTION_NAME)
        return run_agent_loop(
            question,
            api_client=model_client,
            search_fn=lambda query: execute_search(query, collection),
            max_tool_rounds=max_tool_rounds,
        )


def main() -> int:
    try:
        Config.require_api_key("SILICONFLOW_API_KEY")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1
    test_cases = [
        ("OB-4.2.1 版本和旧版本兼容吗？", "需要检索（版本兼容性）"),
        ("遇到 E-4012 错误怎么解决？", "需要检索（错误码）"),
        ("2024年Q3的总营收是多少？", "需要检索（财务数据）"),
        ("你好，今天天气怎么样？", "不需要检索（闲聊）"),
    ]

    print("=" * 60)
    print("Agentic RAG 演示")
    print("=" * 60)
    with create_model_client() as model_client:
        for index, (question, expected) in enumerate(test_cases, 1):
            print(f"\n【问题 {index}】{question}")
            print(f"  预期行为：{expected}")
            answer = ask_agent(question, api_client=model_client)
            print(f"  回答：{answer[:200]}{'...' if len(answer) > 200 else ''}")
            print()

    print("=" * 60)
    print("✅ Agentic RAG 演示完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
