"""Deterministic offline Knowledge Agent used by the P5 evaluation examples."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import AgentResult, KnowledgeDocument


DEFAULT_KNOWLEDGE_BASE = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


class KnowledgeAgent:
    """A rule-based Knowledge Agent that mimics key production behaviors offline.

    这个类刻意只负责“读取知识库 -> 判断命中 -> 生成结构化结果”。
    评测指标和报告生成放在 evaluation 模块，避免把业务运行逻辑和度量逻辑混在一起。
    """

    def __init__(self, knowledge_base_path: str | Path = DEFAULT_KNOWLEDGE_BASE) -> None:
        # 初始化时一次性读取知识库，模拟真实 Agent 启动时加载索引或配置的过程。
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = self._load_documents(self.knowledge_base_path)

    @staticmethod
    def _load_documents(path: Path) -> tuple[KnowledgeDocument, ...]:
        """读取本地 JSON 知识库，并转换为后续规则匹配使用的 KnowledgeDocument。"""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"cannot read knowledge base: {path}") from exc
        if not isinstance(raw, list):
            raise ValueError("knowledge base root must be a list")
        return tuple(KnowledgeDocument.from_mapping(item) for item in raw)

    def run(self, query: str, task_id: str | None = None) -> AgentResult:
        """Run one offline query and return a stable, evaluation-friendly result.

        当前阶段用规则模拟而不调用真实 LLM，是为了先固定业务指标口径和报告结构：
        这样测试不受网络、密钥、模型版本和随机采样影响，后续替换真实模型时也能复用同一套 evaluator。
        """
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()

        # 第一优先级：模拟检索系统异常。它不是“没有答案”，而是基础设施暂时无法给出依据，
        # 因此正确业务动作是转人工，而不是继续尝试生成答案。
        if self._is_retrieval_failure(lowered_query):
            return self._build_result(
                answer="知识检索服务暂时不可用，已转人工处理。",
                retrieval_hit=False,
                retrieval_failed=True,
                knowledge_available=False,
                tool_called=False,
                tool_success=False,
                handoff=True,
                task_success=False,
                hallucinated=False,
                matched_doc_id=None,
                category="exception",
                query=normalized_query,
            )

        # 检索漏召回与检索服务异常、知识缺失是三个不同状态：这里明确知道知识库有 SLA，
        # 但模拟检索器没有返回它，以便离线评测和线上监控都能复现 false negative。
        if self._is_retrieval_miss(lowered_query):
            return self._build_result(
                answer="未检索到足够依据，已转人工处理。",
                retrieval_hit=False,
                retrieval_failed=False,
                knowledge_available=True,
                tool_called=False,
                tool_success=False,
                handoff=True,
                task_success=False,
                hallucinated=False,
                matched_doc_id=None,
                category="general",
                query=normalized_query,
            )

        # 第二优先级：保留一个明确的幻觉样本。它模拟模型在没有知识支撑时仍然给出确定答案，
        # 这类风险需要在评测阶段被单独统计出来，不能被普通回答准确率掩盖。
        if self._is_hallucination_case(lowered_query):
            return self._build_result(
                answer="QuantumX 9000 支持自动量子归档并提供终身免费升级。",
                retrieval_hit=False,
                retrieval_failed=False,
                knowledge_available=False,
                tool_called=False,
                tool_success=False,
                handoff=False,
                task_success=False,
                hallucinated=True,
                matched_doc_id=None,
                category="exception",
                query=normalized_query,
            )

        # 第三优先级：正常知识检索。先拿到文档，再决定是否需要调用工具或转人工。
        document = self._match_document(normalized_query)
        if document is None:
            # 知识缺失时直接转人工，业务上表示 Agent 不应为了完成对话而编造答案。
            return self._build_result(
                answer="当前知识库没有覆盖该问题，已转人工处理。",
                retrieval_hit=False,
                retrieval_failed=False,
                knowledge_available=False,
                tool_called=False,
                tool_success=False,
                handoff=True,
                task_success=False,
                hallucinated=False,
                matched_doc_id=None,
                category="missing",
                query=normalized_query,
            )

        tool_called = document.requires_tool
        tool_success = self._simulate_tool_success(document, lowered_query)
        # 只有“确实调用了工具且工具失败”才触发这里的转人工；普通知识问答不受工具状态影响。
        handoff = tool_called and not tool_success
        answer = document.answer
        if tool_called and not tool_success:
            answer = f"{answer} 工具执行失败，已转人工处理。"

        return self._build_result(
            answer=answer,
            retrieval_hit=True,
            retrieval_failed=False,
            knowledge_available=True,
            tool_called=tool_called,
            tool_success=tool_success,
            handoff=handoff,
            task_success=not handoff,
            hallucinated=False,
            matched_doc_id=document.doc_id,
            category=document.category,
            query=normalized_query,
        )

    def _match_document(self, query: str) -> KnowledgeDocument | None:
        """按业务风险从高到低匹配知识库文档。"""
        lowered_query = query.lower()
        # 错误码通常是最明确的故障定位信号；只要 query 包含错误码，就直接返回对应文档。
        for document in self.documents:
            if document.error_code and document.error_code.lower() in lowered_query:
                return document
        # 产品型号和版本需要精确匹配，避免把 X100 v1.2 的限制误答成 v2.0 的能力。
        for document in self.documents:
            if document.product_model and document.product_model.lower() in lowered_query:
                if document.product_version is None or document.product_version.lower() in lowered_query:
                    return document

        # 普通知识检索用简单关键词打分模拟。这里不是要实现搜索引擎，而是用可解释的规则
        # 表达“命中本地知识”的概念；分数大于 0 才算检索命中。
        best: tuple[int, KnowledgeDocument] | None = None
        for document in self.documents:
            score = sum(1 for keyword in document.keywords if keyword.lower() in lowered_query)
            if score > 0 and (best is None or score > best[0]):
                best = (score, document)
        return None if best is None else best[1]

    @staticmethod
    def _simulate_tool_success(document: KnowledgeDocument, lowered_query: str) -> bool:
        """模拟外部工具调用结果，不真实访问订单、工单或账号系统。"""
        # 工具调用成功率在真实系统中来自外部 API；这里用文档配置和 query 标记模拟可复现的成功/失败路径。
        # 如果文档不需要工具，返回 False 表示“没有成功的工具调用”，而不是表示任务失败。
        if not document.requires_tool:
            return False
        # 允许测试或数据集通过 query 标记强制失败，便于覆盖异常分支。
        if "工具失败" in lowered_query or "fail_tool" in lowered_query:
            return False
        return document.tool_success

    @staticmethod
    def _is_retrieval_failure(lowered_query: str) -> bool:
        """识别检索基础设施失败样本。"""
        # 检索失败代表基础设施或索引异常，不等同于知识库没有答案。
        # 这类问题通常需要告警和人工兜底，后续阶段可映射到运行层监控指标。
        return "检索失败" in lowered_query or "simulate_retrieval_failure" in lowered_query

    @staticmethod
    def _is_retrieval_miss(lowered_query: str) -> bool:
        """识别知识存在、检索完成但未返回文档的可复现漏召回样本。"""
        return "simulate_retrieval_miss" in lowered_query

    @staticmethod
    def _is_hallucination_case(lowered_query: str) -> bool:
        """识别专门用于评测的幻觉样本。"""
        # 幻觉样本用于验证 evaluator 能识别“没有知识却给出确定答案”的高风险行为。
        # 这里故意不转人工，是为了让报告中能看到 hallucination_rate 被正确拉高。
        return "幻觉" in lowered_query or "quantumx" in lowered_query

    @staticmethod
    def _build_result(
        *,
        answer: str,
        retrieval_hit: bool,
        retrieval_failed: bool,
        knowledge_available: bool,
        tool_called: bool,
        tool_success: bool,
        handoff: bool,
        task_success: bool,
        hallucinated: bool,
        matched_doc_id: str | None,
        category: str,
        query: str,
    ) -> AgentResult:
        """集中构造 AgentResult，保证所有分支都返回同一套字段。"""
        # token、成本和延迟是确定性估算值，只用于教程阶段跑通监控口径；
        # 等接入真实模型后，可以把这些字段替换为 provider 返回的真实 usage 和 tracing 数据。
        token_usage = max(20, (len(query) + len(answer)) // 2)
        latency_ms = 45 + len(query) * 3 + (35 if tool_called else 0) + (25 if handoff else 0)
        return AgentResult(
            answer=answer,
            retrieval_hit=retrieval_hit,
            retrieval_failed=retrieval_failed,
            knowledge_available=knowledge_available,
            tool_called=tool_called,
            tool_success=tool_success,
            handoff=handoff,
            task_success=task_success,
            hallucinated=hallucinated,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost=round(token_usage * 0.00002, 6),
            matched_doc_id=matched_doc_id,
            category=category,
        )
