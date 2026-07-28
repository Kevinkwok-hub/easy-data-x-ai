"""
D2 高级 Chunking 策略实现

包含三种策略：
  1. semantic_chunk       — 语义分块（基于句子 Embedding 的断点检测）
  2. dynamic_overlap_chunk — 动态 overlap（边界处加大重叠）
  3. parent_child_chunk   — 父子分块（小 chunk 检索，大 chunk 返回上下文）

另提供 fixed_overlap_chunk 作为对比基线。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Real


@dataclass
class ChildChunk:
    """子 chunk：用于向量索引的细粒度片段。"""

    text: str
    parent_id: str
    child_index: int


@dataclass
class ParentChunk:
    """父 chunk：检索命中后返回给 LLM 的上下文块。"""

    parent_id: str
    text: str
    children: list[ChildChunk] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    """按中英文句号、问号、感叹号切分句子。"""
    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed_overlap_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[str]:
    """固定大小 + 固定 overlap 的基线分块策略。"""
    _validate_fixed_chunk_parameters(chunk_size, overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def _validate_fixed_chunk_parameters(chunk_size: int, overlap: int) -> None:
    """校验固定分块参数，确保循环步长始终为正数。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_semantic_breakpoints(
    similarities: list[float],
    percentile: float = 95.0,
) -> set[int]:
    """
    在相邻句相似度序列中，找出「语义跳变」断点。
    采用 LangChain SemanticChunker 的 percentile 思路：
    当相似度跌幅超过第 percentile 分位时切分。
    """
    if len(similarities) < 2:
        return set()

    drops = [1.0 - sim for sim in similarities]
    if max(drops) == min(drops):
        return set()

    threshold = _percentile(drops, percentile)

    breakpoints: set[int] = set()
    for idx, drop in enumerate(drops):
        if drop >= threshold:
            breakpoints.add(idx + 1)
    return breakpoints


def _percentile(values: list[float], percentile: float) -> float:
    """用线性插值计算 percentile，避免小样本时阈值过度偏向最大值。"""
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    bounded = min(max(percentile, 0.0), 100.0)
    position = (len(sorted_values) - 1) * bounded / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    ratio = position - lower
    return sorted_values[lower] * (1 - ratio) + sorted_values[upper] * ratio


def semantic_chunk(
    text: str,
    embed_fn,
    percentile: float = 95.0,
    min_chunk_chars: int = 80,
    max_chunk_chars: int = 600,
) -> list[str]:
    """
    语义分块：逐句 Embedding，在主题跳变处插入断点。

    embed_fn: 接收字符串列表，返回等长向量列表的函数。
    参考 NAACL 2025 Findings《Is Semantic Chunking Worth the Computational Cost?》
    与 LangChain SemanticChunker 的 percentile 阈值实现。
    """
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, Real)
        or not math.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile 必须是 0 到 100 之间的有限数值")
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars 必须大于 0")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars 必须大于 0")
    if min_chunk_chars > max_chunk_chars:
        raise ValueError("min_chunk_chars 不能大于 max_chunk_chars")

    sentences = split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        raw_chunks = sentences
    else:
        embeddings = list(embed_fn(sentences))
        _validate_embeddings(embeddings, len(sentences))
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        breakpoints = _find_semantic_breakpoints(similarities, percentile)

        raw_chunks = []
        start = 0
        for idx in range(1, len(sentences) + 1):
            if idx in breakpoints or idx == len(sentences):
                raw_chunks.append("".join(sentences[start:idx]))
                start = idx

    return _merge_small_chunks(raw_chunks, min_chunk_chars, max_chunk_chars)


def _validate_embeddings(embeddings: list[list[float]], expected_count: int) -> None:
    """校验 Embedding 数量、维度及数值有效性。"""
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding 数量应为 {expected_count}，实际为 {len(embeddings)}"
        )

    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        raise ValueError("Embedding 向量不能为空")

    for vector in embeddings:
        if len(vector) != dimension:
            raise ValueError("Embedding 向量维度必须一致")
        try:
            finite = all(math.isfinite(value) for value in vector)
        except TypeError as exc:
            raise ValueError("Embedding 向量必须只包含数值") from exc
        if not finite:
            raise ValueError("Embedding 向量必须只包含有限数值")


def _merge_small_chunks(
    chunks: list[str],
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """合并过小的语义块，并保证每个块不超过最大长度。"""
    bounded_chunks = [
        chunk[start:start + max_chars]
        for chunk in chunks
        for start in range(0, len(chunk), max_chars)
    ]
    merged: list[str] = []
    buffer = ""
    for chunk in bounded_chunks:
        candidate = buffer + chunk
        if len(candidate) <= max_chars and len(candidate) < min_chars:
            buffer = candidate
            continue
        if len(candidate) > max_chars:
            merged.append(buffer)
            buffer = chunk
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        if merged and len(merged[-1]) + len(buffer) <= max_chars:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return [c.strip() for c in merged if c.strip()]


def _detect_boundary_positions(text: str) -> set[int]:
    """标记段落、标题、表格和代码块等需要加大 overlap 的位置。"""
    boundaries: set[int] = set()
    for match in re.finditer(r"\n\n+", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^#{1,3}\s", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^[ \t]*\|.+\|[ \t]*$", text):
        boundaries.add(match.start())
    for match in re.finditer(r"(?m)^```", text):
        boundaries.add(match.start())
    return boundaries


def _overlap_near_boundary(
    position: int,
    boundaries: set[int],
    base_overlap: int,
    max_overlap: int,
    window: int = 40,
) -> int:
    """边界附近使用更大 overlap，同质段落内使用较小 overlap。"""
    for boundary in boundaries:
        if abs(position - boundary) <= window:
            return max_overlap
    return base_overlap


def dynamic_overlap_chunk(
    text: str,
    chunk_size: int = 200,
    base_overlap: int = 20,
    max_overlap: int = 60,
) -> list[str]:
    """
    动态 overlap：在段落、标题、表格和代码块边界处加大重叠，
    同质段落内减小重叠。

    思路参考 Weaviate《Chunking Strategies for RAG》中的 Adaptive Chunking：
    根据内容结构动态调整参数，而非全文统一 overlap。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if base_overlap < 0 or max_overlap < 0:
        raise ValueError("overlap 必须大于等于 0")
    if base_overlap > max_overlap:
        raise ValueError("base_overlap 不能大于 max_overlap")
    if max_overlap >= chunk_size:
        raise ValueError("max_overlap 必须小于 chunk_size")

    if not text.strip():
        return []

    boundaries = _detect_boundary_positions(text)
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        overlap = _overlap_near_boundary(end, boundaries, base_overlap, max_overlap)
        stride = max(chunk_size - overlap, 1)
        start += stride

    return chunks


def parent_child_chunk(
    text: str,
    parent_size: int = 500,
    child_size: int = 120,
    child_overlap: int = 20,
) -> list[ParentChunk]:
    """
    父子分块：父块保留完整上下文，子块用于精准检索。

    检索时只索引子块；命中后通过 parent_id 取回父块文本给 LLM。
    又称 small-to-big / parent-document retrieval。
    """
    _validate_fixed_chunk_parameters(parent_size, 0)
    _validate_fixed_chunk_parameters(child_size, child_overlap)

    parent_texts = fixed_overlap_chunk(text, parent_size, overlap=0)
    parents: list[ParentChunk] = []

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_idx}"
        child_texts = fixed_overlap_chunk(parent_text, child_size, child_overlap)
        children = [
            ChildChunk(
                text=child_text,
                parent_id=parent_id,
                child_index=child_idx,
            )
            for child_idx, child_text in enumerate(child_texts)
        ]
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=parent_text,
                children=children,
            )
        )
    return parents


def flatten_parent_child(parents: list[ParentChunk]) -> tuple[list[str], list[dict]]:
    """将父子结构展平为 seekdb 可写入的 ids / texts / metadatas。"""
    texts: list[str] = []
    metadatas: list[dict] = []
    for parent in parents:
        for child in parent.children:
            texts.append(child.text)
            metadatas.append(
                {
                    "parent_id": parent.parent_id,
                    "parent_text": parent.text,
                    "child_index": child.child_index,
                    "chunk_role": "child",
                }
            )
    return texts, metadatas
