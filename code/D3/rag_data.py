"""D3 示例知识库的纯数据加载器，不导入数据库或模型依赖。"""

from __future__ import annotations

import json
from pathlib import Path


KNOWLEDGE_PATH = Path(__file__).resolve().parent / "data" / "knowledge_base.json"


def load_knowledge_chunks(path: Path = KNOWLEDGE_PATH) -> list[dict[str, str]]:
    """加载并校验知识片段的必填字段和唯一 ID。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("知识库数据必须是非空 JSON 数组")

    required_fields = {"id", "content", "doc_type", "version"}
    seen_ids: set[str] = set()
    chunks: list[dict[str, str]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or not required_fields.issubset(item):
            raise ValueError(f"第 {index} 条知识片段缺少必填字段")
        normalized = {
            field: str(item[field]).strip()
            for field in required_fields
        }
        if not all(normalized.values()):
            raise ValueError(f"第 {index} 条知识片段包含空字段")
        if normalized["id"] in seen_ids:
            raise ValueError(f"知识片段 ID 重复：{normalized['id']}")
        seen_ids.add(normalized["id"])
        chunks.append(normalized)
    return chunks


knowledge_chunks = load_knowledge_chunks()
