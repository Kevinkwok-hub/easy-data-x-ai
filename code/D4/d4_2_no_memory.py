import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from openai import OpenAI

# ============================================================
# d4_2：无记忆 Agent 演示
#
# 演示：
#   1. 每次对话都是全新的，不保留任何上下文
#   2. Agent 无法记住用户的身份、偏好、历史信息
#   3. 为 d4_3（有记忆版本）提供对比基准
#
# 运行：python d4_2_no_memory.py
# ============================================================


# ---------- 1. 初始化 LLM 客户端 ----------

MODEL = "deepseek-ai/DeepSeek-V3"


def create_model_client():
    """显式创建 OpenAI 兼容模型客户端。"""
    return OpenAI(
        api_key=Config.SILICONFLOW_API_KEY,
        base_url=Config.SILICONFLOW_BASE_URL,
    )


def _require_response_content(response) -> str:
    """把供应商空响应转换为稳定异常。"""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("模型返回空响应：缺少 choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("模型返回空响应：缺少 message")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型返回空响应：缺少 content")
    return content


# ---------- 2. 无记忆的 Agent ----------

def chat_without_memory(user_input: str, *, api_client=None) -> str:
    """
    无记忆的 Agent：每次对话都是全新的。
    messages 列表只包含当前这一轮的 system prompt 和用户输入，
    没有任何历史信息。
    """
    if api_client is None:
        with create_model_client() as owned_client:
            return chat_without_memory(user_input, api_client=owned_client)
    messages = [
        {
            "role": "system",
            "content": "你是一个友好的技术助手。根据用户的问题提供有针对性的建议。"
        },
        {"role": "user", "content": user_input}
    ]

    active_client = api_client if api_client is not None else create_model_client()
    response = active_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return _require_response_content(response)


# ---------- 3. 演示对话序列 ----------

def run_demo(*, api_client=None):
    """运行四轮无记忆对话演示。"""
    active_client = api_client if api_client is not None else create_model_client()
    print("=" * 60)
    print("无记忆 Agent 演示")
    print("=" * 60)
    print("观察：每一轮对话都是独立的，Agent 不记得上一轮说过什么")
    print()

    questions = (
        ("【第 1 轮】告知身份和偏好", "我是一个 Python 开发者，主要做后端开发，喜欢简洁的回答。"),
        ("【第 2 轮】推荐 Web 框架", "帮我推荐一个 Web 框架"),
        ("【第 3 轮】询问缓存方案", "怎么给我的项目加缓存？"),
        ("【第 4 轮】模拟重启后继续昨天的话题", "继续昨天的话题，帮我选一个数据库方案。"),
    )
    for title, question in questions:
        print(f"\n{title}")
        print("-" * 40)
        print(f"用户：{question}")
        answer = chat_without_memory(question, api_client=active_client)
        print(f"Agent：{answer[:300]}{'...' if len(answer) > 300 else ''}")

    print()
    print("=" * 60)
    print("总结：无记忆 Agent 的问题")
    print("  1. 每轮对话独立，无法利用用户的历史信息")
    print("  2. 推荐不够个性化，无法针对用户的技术栈")
    print("  3. 重启后完全失忆，用户需要重复介绍自己")
    print("  → 运行 d4_3_with_memory.py 查看有记忆版本的效果对比")


def main() -> int:
    try:
        Config.require_api_key("SILICONFLOW_API_KEY")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1
    with create_model_client() as model_client:
        run_demo(api_client=model_client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
