"""
d4_6：长期记忆压缩策略

演示：
  1. 写入时压缩：对话原文 vs 事实化入库的体积对比
  2. 存续中压缩：多条弱信号巩固蒸馏成一条稳定事实
  3. 结构压缩：用户画像（固定字段）与长尾事实库分存
  4. 成本对照：Top-5 拼进 Prompt 时的上下文 token 预算

对应课程：D4 第六部分「压缩策略：先少存，再好存」
对应 Issue：datawhalechina/easy-data-x-ai#32

运行：
  python d4_6_memory_compression.py

无需外部 API / seekdb，纯 Python 即可跑通。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------- 1. 模拟对话与规则提炼（替代 LLM，保证可离线断言）----------

DIALOGUE_ROUNDS: list[tuple[str, str]] = [
    (
        "我是一个 Python 开发者，在一家做 SaaS 的创业公司做后端。",
        "了解了，你是 Python 后端，后续推荐会优先 Python 生态。",
    ),
    (
        "我喜欢简洁的回答，不需要太多铺垫。",
        "好的，之后我会尽量给结论和关键步骤。",
    ),
    (
        "我们公司用的是 AWS。",
        "记下了，部署相关建议会默认围绕 AWS。",
    ),
    (
        "我最近还在用 Flask，感觉有点慢。",
        "Flask 轻量但高并发场景可能吃力，可以评估异步框架。",
    ),
    (
        "准备迁 FastAPI。",
        "FastAPI 适合异步和类型提示，迁移成本通常可控。",
    ),
    (
        "FastAPI 已经上手了，旧项目还留着一点 Flask。",
        "主栈切到 FastAPI 后，可以把遗留 Flask 模块逐步替换。",
    ),
    (
        "团队淘汰过 RabbitMQ，觉得运维成本高。",
        "可以看看托管队列或更轻量的方案，减少自建运维。",
    ),
    (
        "曾用 FastAPI + Redis 做过会话缓存，效果不错。",
        "会话热数据放 Redis 是常见且有效的组合。",
    ),
    (
        "帮我推荐一个消息队列方案。",
        "结合你的偏好，建议先看 AWS SQS 或 Redis Stream。",
    ),
    (
        "数据库怎么选？",
        "Python + AWS 场景可优先 PostgreSQL，或评估 seekdb。",
    ),
]


# 关键词 → 事实：模拟 LLM 提炼，覆盖课程叙述中的关键事实
FACT_RULES: list[tuple[list[str], str]] = [
    (["python", "后端"], "用户是 Python 后端开发者"),
    (["简洁"], "用户偏好简洁回答"),
    (["aws"], "用户公司使用 AWS"),
    (["flask", "慢"], "用户认为 Flask 偏慢，准备迁移"),
    (["迁 fastapi", "准备迁"], "用户准备迁移到 FastAPI"),
    (["上手", "fastapi"], "用户后端已切到 FastAPI，仍有少量 Flask 遗留"),
    (["rabbitmq"], "团队淘汰过 RabbitMQ，觉得运维成本高"),
    (["redis", "会话"], "曾用 FastAPI + Redis 做过会话缓存"),
]


PROFILE_RULES: list[tuple[list[str], str, str]] = [
    (["python"], "primary_language", "Python"),
    (["后端"], "role", "backend"),
    (["aws"], "cloud", "AWS"),
    (["简洁"], "answer_style", "concise"),
]


def estimate_tokens(text: str) -> int:
    """粗估 token：中英混排用字符数 / 2，足够做相对对比"""
    return max(1, (len(text) + 1) // 2)


def extract_facts(user_input: str, assistant_reply: str) -> list[str]:
    """规则版事实提炼：命中关键词则产出结构化事实"""
    blob = f"{user_input} {assistant_reply}".lower()
    facts: list[str] = []
    for keywords, fact in FACT_RULES:
        if all(kw.lower() in blob for kw in keywords):
            if fact not in facts:
                facts.append(fact)
    return facts


def extract_profile_updates(user_input: str) -> dict[str, str]:
    """从对话中抽出可覆盖写的画像字段"""
    lower = user_input.lower()
    updates: dict[str, str] = {}
    for keywords, key, value in PROFILE_RULES:
        if any(kw.lower() in lower for kw in keywords):
            updates[key] = value
    return updates


# ---------- 2. 两种写入方案的存储 ----------

@dataclass
class RawMemoryStore:
    """坏做法：把对话原文整段入库"""

    records: list[str] = field(default_factory=list)

    def add_transcript(self, user_input: str, assistant_reply: str) -> None:
        self.records.append(f"用户: {user_input}\n助手: {assistant_reply}")

    def total_chars(self) -> int:
        return sum(len(item) for item in self.records)

    def total_tokens(self) -> int:
        return sum(estimate_tokens(item) for item in self.records)

    def top_k_context_tokens(self, k: int = 5) -> int:
        # 原文方案没有排序能力时，取最近 k 条模拟 Prompt 注入
        selected = self.records[-k:]
        return sum(estimate_tokens(item) for item in selected)


@dataclass
class FactMemoryStore:
    """好做法：只存提炼后的事实；支持按话题巩固蒸馏"""

    facts: list[str] = field(default_factory=list)
    archived_fragments: list[str] = field(default_factory=list)

    def add_facts(self, new_facts: list[str]) -> None:
        for fact in new_facts:
            if fact not in self.facts:
                self.facts.append(fact)

    def total_chars(self) -> int:
        return sum(len(item) for item in self.facts)

    def total_tokens(self) -> int:
        return sum(estimate_tokens(item) for item in self.facts)

    def top_k_context_tokens(self, k: int = 5) -> int:
        selected = self.facts[:k]
        return sum(estimate_tokens(item) for item in selected)

    def consolidate_topic(self, topic_keywords: list[str], min_count: int = 3) -> str | None:
        """
        把同一话题下的多条碎片记忆，蒸馏成一条稳定事实。
        对应课程伪代码 consolidate_topic。
        """
        matched = [
            fact for fact in self.facts
            if any(kw.lower() in fact.lower() for kw in topic_keywords)
        ]
        if len(matched) < min_count:
            return None

        distilled = "用户后端已从 Flask 迁移到 FastAPI"
        # 碎片移出检索面，只保留稳定事实
        self.archived_fragments.extend(matched)
        self.facts = [fact for fact in self.facts if fact not in matched]
        if distilled not in self.facts:
            self.facts.insert(0, distilled)
        return distilled


@dataclass
class ProfileFactStore:
    """结构压缩：画像覆盖写 + 事实库增量追加"""

    profile: dict[str, str] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)

    def upsert_profile(self, updates: dict[str, str]) -> None:
        self.profile.update(updates)

    def add_fact(self, fact: str) -> None:
        # 已进入画像的稳定字段，不再重复堆进事实库
        profile_values = {value.lower() for value in self.profile.values()}
        if any(value in fact.lower() for value in profile_values if len(value) >= 3):
            # 仍允许长尾细节入库；仅跳过“纯画像复述”
            if fact.startswith("用户是") or fact.startswith("用户偏好") or "公司使用" in fact:
                return
        if fact not in self.facts:
            self.facts.append(fact)

    def query_combined(self, query: str) -> dict[str, Any]:
        """画像给确定性约束（热路径始终可读），事实库按需补长尾"""
        deterministic = dict(self.profile)
        keywords = [kw for kw in query.lower().split() if len(kw) >= 2]
        contextual = [
            fact for fact in self.facts
            if any(kw in fact.lower() for kw in keywords)
        ]
        return {"deterministic": deterministic, "contextual": contextual}


# ---------- 3. 演示与断言 ----------

def demo_write_time_compression() -> tuple[RawMemoryStore, FactMemoryStore]:
    print("=" * 64)
    print("实验 1：写入时压缩（原文 vs 事实化）")
    print("=" * 64)

    raw_store = RawMemoryStore()
    fact_store = FactMemoryStore()

    for user_input, assistant_reply in DIALOGUE_ROUNDS:
        raw_store.add_transcript(user_input, assistant_reply)
        fact_store.add_facts(extract_facts(user_input, assistant_reply))

    raw_tokens = raw_store.total_tokens()
    fact_tokens = fact_store.total_tokens()
    ratio = raw_tokens / fact_tokens if fact_tokens else float("inf")

    print(f"\n  对话轮数：{len(DIALOGUE_ROUNDS)}")
    print(f"  原文方案：{len(raw_store.records)} 条，约 {raw_tokens} tokens")
    print(f"  事实方案：{len(fact_store.facts)} 条，约 {fact_tokens} tokens")
    print(f"  压缩比 ≈ {ratio:.1f}:1")
    print("\n  事实条目：")
    for fact in fact_store.facts:
        print(f"    - {fact}")

    assert len(fact_store.facts) < len(raw_store.records)
    assert fact_tokens * 3 < raw_tokens, "fact store should be much smaller"
    assert ratio >= 3, "expected meaningful compression ratio"

    raw_ctx = raw_store.top_k_context_tokens(5)
    fact_ctx = fact_store.top_k_context_tokens(5)
    print(f"\n  Top-5 Prompt 预算：原文 {raw_ctx} tokens vs 事实 {fact_ctx} tokens")
    assert fact_ctx < raw_ctx
    print("  → 正确：事实化同时省存储和上下文成本")
    return raw_store, fact_store


def demo_consolidation(fact_store: FactMemoryStore) -> None:
    print("\n" + "=" * 64)
    print("实验 2：存续中压缩（巩固蒸馏）")
    print("=" * 64)

    before = len(fact_store.facts)
    print(f"\n  蒸馏前事实数：{before}")
    distilled = fact_store.consolidate_topic(
        # 只聚合成“迁移”话题，避免把 Redis 等长尾事实误归档
        topic_keywords=["Flask", "偏慢", "迁移到 FastAPI", "切到 FastAPI"],
        min_count=3,
    )
    print(f"  蒸馏结果：{distilled}")
    print(f"  蒸馏后事实数：{len(fact_store.facts)}")
    print(f"  归档碎片数：{len(fact_store.archived_fragments)}")

    assert distilled == "用户后端已从 Flask 迁移到 FastAPI"
    assert distilled in fact_store.facts
    assert len(fact_store.facts) < before
    assert len(fact_store.archived_fragments) >= 3
    print("  → 正确：多条弱信号变成一条稳定事实，碎片退出检索面")


def demo_profile_fact_split() -> None:
    print("\n" + "=" * 64)
    print("实验 3：结构压缩（画像 + 长尾事实分存）")
    print("=" * 64)

    store = ProfileFactStore()
    for user_input, assistant_reply in DIALOGUE_ROUNDS:
        store.upsert_profile(extract_profile_updates(user_input))
        for fact in extract_facts(user_input, assistant_reply):
            store.add_fact(fact)

    print("\n  用户画像（固定字段，覆盖写）：")
    for key, value in store.profile.items():
        print(f"    - {key}: {value}")
    print(f"\n  长尾事实库（{len(store.facts)} 条）：")
    for fact in store.facts:
        print(f"    - {fact}")

    assert store.profile.get("primary_language") == "Python"
    assert store.profile.get("cloud") == "AWS"
    assert store.profile.get("answer_style") == "concise"
    assert any("RabbitMQ" in fact for fact in store.facts)
    assert any("Redis" in fact for fact in store.facts)

    result = store.query_combined("推荐后端部署方案 AWS")
    print("\n  [组合查询] 推荐后端部署方案 AWS")
    print(f"    画像约束：{result['deterministic']}")
    print(f"    长尾补充：{result['contextual']}")
    assert result["deterministic"]["cloud"] == "AWS"
    assert result["deterministic"]["primary_language"] == "Python"
    print("  → 正确：热路径读画像，向量/事实库只补长尾")


def print_summary() -> None:
    print("\n" + "=" * 64)
    print("总结")
    print("  1. 写入时只存事实，不把对话原文堆进检索面")
    print("  2. 巩固蒸馏用「多条变一条」换压缩比，而不是把字压扁")
    print("  3. 画像承载稳定约束，事实库承载长尾细节")
    print("  4. 压缩省下的不只是磁盘，还有每次回答的 Prompt 预算")
    print("=" * 64)


def main() -> None:
    _, fact_store = demo_write_time_compression()
    demo_consolidation(fact_store)
    demo_profile_fact_split()
    print_summary()
    print("\n[PASS] d4_6_memory_compression 全部断言通过")


if __name__ == "__main__":
    main()
