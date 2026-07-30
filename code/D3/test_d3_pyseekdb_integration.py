import contextlib
import io
import os
import platform
import runpy
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import pyseekdb
from pyseekdb.client.embedding_function import register_embedding_function


D3_DIR = Path(__file__).resolve().parent


def wait_for_documents(
    search,
    timeout_seconds=10,
    expected_top1_contains=None,
):
    """索引可能异步就绪，等待非空且目标 Top-1 真正可用。"""
    deadline = time.monotonic() + timeout_seconds
    last_result = {}
    while time.monotonic() < deadline:
        last_result = search()
        documents = last_result.get("documents", [[]])
        has_documents = documents and documents[0]
        top1_is_ready = (
            expected_top1_contains is None
            or (
                has_documents
                and expected_top1_contains in documents[0][0]
            )
        )
        if has_documents and top1_is_ready:
            return last_result
        time.sleep(0.1)
    raise AssertionError(
        f"seekdb 索引在 {timeout_seconds} 秒内未返回预期文档：{last_result}"
    )


@register_embedding_function
class LocalEmbedding:
    """离线测试向量模型，不访问任何外部 API。"""

    dimension = 4

    @staticmethod
    def name():
        return "d3_test_local_embedding"

    def __call__(self, documents):
        if isinstance(documents, str):
            documents = [documents]
        return [self._embed(document) for document in documents]

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(_config):
        return LocalEmbedding()

    @staticmethod
    def _embed(document):
        text = document.upper()
        vector = [
            float("E-4012" in text),
            float("Q3" in text),
            float("性能" in text),
            0.1,
        ]
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector]


def load_module(file_name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(
            str(D3_DIR / file_name),
            run_name=f"d3_integration_{file_name}",
        )


def create_test_client(temp_dir):
    """优先连接外部测试库；未配置时回退到临时 Embedded 数据库。"""
    host = os.getenv("SEEKDB_TEST_HOST")
    if not host:
        return pyseekdb.Client(path=str(Path(temp_dir) / "seekdb"))
    database = os.getenv("SEEKDB_TEST_DATABASE", "").strip()
    if not database:
        raise RuntimeError("外部集成测试必须显式配置 SEEKDB_TEST_DATABASE")
    if os.getenv("SEEKDB_ALLOW_DESTRUCTIVE") != "1":
        raise RuntimeError("外部集成测试必须设置 SEEKDB_ALLOW_DESTRUCTIVE=1")
    return pyseekdb.Client(
        host=host,
        port=int(os.getenv("SEEKDB_TEST_PORT", "2881")),
        user=os.getenv("SEEKDB_TEST_USER", "root"),
        password=os.getenv("SEEKDB_TEST_PASSWORD", ""),
        database=database,
    )


class WaitForDocumentsTests(unittest.TestCase):
    def test_waits_past_transient_top1_until_expected_document_is_ready(self):
        responses = iter(
            [
                {"documents": [["临时返回的其他文档"]]},
                {"documents": [["错误码 E-4012：数据库连接池耗尽。"]]},
            ]
        )

        result = wait_for_documents(
            lambda: next(responses),
            timeout_seconds=1,
            expected_top1_contains="E-4012",
        )

        self.assertIn("E-4012", result["documents"][0][0])


class RealPyseekdbIntegrationTests(unittest.TestCase):
    def test_ingest_query_hybrid_and_upsert_in_temporary_database(self):
        if platform.system() == "Darwin" and not os.getenv("SEEKDB_TEST_HOST"):
            self.skipTest(
                "macOS 不支持 Embedded seekdb；请配置 SEEKDB_TEST_HOST 连接测试服务"
            )
        ingest = load_module("d3_1_ingest.py")
        compare = load_module("d3_3_compare.py")
        production = load_module("d3_4_production.py")
        collection_name = f"d3_product_kb_test_{uuid.uuid4().hex}"
        ingest["build_knowledge_base"].__globals__["COLLECTION_NAME"] = collection_name

        with tempfile.TemporaryDirectory() as temp_dir:
            collection_created = False
            # 客户端必须在上下文中关闭，避免测试后残留连接和子进程。
            with create_test_client(temp_dir) as database:
                try:
                    database.delete_collection(collection_name)
                except ValueError:
                    pass
                try:
                    collection = ingest["build_knowledge_base"](
                        database,
                        embedding_function=LocalEmbedding(),
                    )
                    collection_created = True
                    self.assertEqual(12, collection.count())

                    vector_results = wait_for_documents(
                        lambda: compare["vector_only"](
                            "E-4012",
                            target_collection=collection,
                        ),
                        expected_top1_contains="E-4012",
                    )
                    self.assertIn("E-4012", vector_results["documents"][0][0])

                    hybrid_results = wait_for_documents(
                        lambda: compare["hybrid_with_keyword"](
                            "E-4012 错误怎么解决",
                            "E-4012",
                            target_collection=collection,
                        ),
                        expected_top1_contains="E-4012",
                    )
                    self.assertIn("E-4012", hybrid_results["documents"][0][0])

                    production["upsert_document"](
                        collection,
                        {
                            "id": "kb_013",
                            "content": "E-4012 新增诊断步骤：先检查连接泄漏。",
                            "doc_type": "error_codes",
                            "version": "4.3.0",
                        },
                    )
                    self.assertEqual(13, collection.count())

                    production["upsert_document"](
                        collection,
                        {
                            "id": "kb_013",
                            "content": "E-4012 更新诊断步骤：检查连接池指标。",
                            "doc_type": "error_codes",
                            "version": "4.3.1",
                        },
                    )
                    self.assertEqual(13, collection.count())
                    updated = collection.get(ids=["kb_013"])
                    self.assertEqual(
                        ["E-4012 更新诊断步骤：检查连接池指标。"],
                        updated["documents"],
                    )
                finally:
                    if collection_created and database.has_collection(collection_name):
                        database.delete_collection(collection_name)


if __name__ == "__main__":
    unittest.main()
