import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from seekdb_runtime import create_seekdb_client, require_destructive_seekdb_access


D1_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = D1_DIR / "seekdb"
COLLECTION_NAME = "d1_agent_kb"
MODEL_NAME = "qwen-plus"
SYSTEM_PROMPT = (
    "你是 AI 技术文档助手。回答 seekdb、RAG、Agent Memory、MCP 问题前，"
    "请先查询知识库；回答使用中文。"
)
DOCS = [
    "seekdb 支持关键词、语义向量和混合检索，混合检索使用 RRF 融合。",
    "Agentic RAG 会主动判断是否检索以及是否需要再次检索。",
    "Agent Memory 可分为语义记忆、情景记忆和程序记忆。",
    "PowerMem 是基于 seekdb 构建的 Agent 记忆系统。",
    "MCP 是连接 AI 工具与外部服务的标准化协议。",
]
DEFAULT_QUESTIONS = [
    "Agentic RAG 和传统 RAG 有什么区别？",
    "Agent 的记忆系统是怎么分类的？有什么工具可以用？",
    "seekdb 支持哪些检索方式？混合检索的原理是什么？",
]


def _close_database(database):
    """兼容 close() 和上下文管理协议两种 pyseekdb 客户端。"""
    close = getattr(database, "close", None)
    if callable(close):
        close()
        return
    exit_method = getattr(database, "__exit__", None)
    if callable(exit_method):
        exit_method(None, None, None)


def build_search_tool(collection):
    @tool
    def search_knowledge_base(query: str) -> str:
        """检索 seekdb、RAG、Agent Memory 和 MCP 相关知识。"""
        results = collection.query(query_texts=[query], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "未找到相关文档。"
        return "\n\n".join(f"[文档{i + 1}] {doc}" for i, doc in enumerate(docs))

    return search_knowledge_base


def run_agent(agent, question: str) -> str:
    final_answer = ""
    for chunk in agent.stream({"messages": [("user", question)]}):
        for node_name, node_output in chunk.items():
            if not isinstance(node_output, dict):
                continue
            for msg in node_output.get("messages", []):
                if getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        print(f"   [行动] 调用工具 '{call['name']}'，参数：{call['args']}")
                elif getattr(msg, "type", None) == "tool":
                    print(f"   [观察] 工具返回了 {msg.content.count('[文档')} 条相关文档")
                elif getattr(msg, "content", None) and node_name == "model":
                    print(msg.content, end="", flush=True)
                    final_answer += msg.content
    print()
    return final_answer


def run_demo(agent, questions=None):
    answers = []
    for index, question in enumerate(questions or DEFAULT_QUESTIONS, 1):
        print(f"\n{'=' * 60}\n问题 {index}：{question}\n{'=' * 60}")
        answer = run_agent(agent, question)
        print(f"\n>>> 最终回答：\n{answer}")
        answers.append(answer)
    return answers


def main(
    model_factory=init_chat_model,
    agent_factory=create_agent,
    client_factory=create_seekdb_client,
    db_path=None,
    questions=None,
) -> int:
    # Key 边界：缺少 Key 时不创建模型，也不触碰本地数据库。
    try:
        Config.require_api_key("DASHSCOPE_API_KEY")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    llm = model_factory(
        MODEL_NAME,
        model_provider="openai",
        **Config.get_dashscope_config(),
    )
    # 用真实轻量调用探测用户已开通的模型；失败时不会创建数据库。
    llm.invoke([("user", "请只回复 ok")])

    # 路径边界：显式路径避免 pyseekdb 使用 cwd，也避免模块级常驻资源。
    resolved_path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    require_destructive_seekdb_access("重建 D1 Agent 知识库")
    db = client_factory(path=str(resolved_path))
    created = False
    try:
        if db.has_collection(COLLECTION_NAME):
            db.delete_collection(COLLECTION_NAME)
        collection = db.create_collection(name=COLLECTION_NAME)
        created = True
        collection.add(
            ids=[f"doc_{i}" for i in range(len(DOCS))],
            documents=DOCS,
        )
        search_tool = build_search_tool(collection)
        agent = agent_factory(
            model=llm,
            tools=[search_tool],
            system_prompt=SYSTEM_PROMPT,
        )
        run_demo(agent, questions)
        return 0
    finally:
        try:
            # 资源清理边界：无论 Agent 是否异常都删除集合，清理异常继续抛出。
            if created and db.has_collection(COLLECTION_NAME):
                db.delete_collection(COLLECTION_NAME)
                print("\n>>> 知识库已清理")
        finally:
            _close_database(db)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(db_path=parse_args().db_path))
