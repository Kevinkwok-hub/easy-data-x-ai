import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


D2_MODULES = (
    "D2.d2_1_ingest",
    "D2.d2_2_vector_search",
    "D2.d2_3_hybrid_search",
    "D2.d2_4_compare",
)


class EmptyCollection:
    def count(self):
        return 0

    def query(self, **_kwargs):
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    def hybrid_search(self, **_kwargs):
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}


class ExistingDatabase:
    def __init__(self, collection=None):
        self.collection = collection or EmptyCollection()

    def has_collection(self, _name):
        return True

    def get_collection(self, _name):
        return self.collection


class ContextOnlyDatabase:
    def __init__(self):
        self.events = []

    def __exit__(self, *_args):
        self.events.append("exit")


class ComparisonCollection(EmptyCollection):
    def __init__(self):
        self.hybrid_calls = []

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return super().hybrid_search(**kwargs)


class D2ImportSafetyTests(unittest.TestCase):
    def test_importing_entry_modules_does_not_open_or_mutate_database(self):
        for module_name in D2_MODULES:
            with self.subTest(module=module_name):
                sys.modules.pop(module_name, None)
                with patch(
                    "pyseekdb.Client",
                    side_effect=AssertionError("导入时不应打开数据库"),
                ):
                    module = importlib.import_module(module_name)

                self.assertTrue(callable(module.main))
                self.assertTrue(callable(module.run_demo))


class D2PathAndOutputTests(unittest.TestCase):
    def test_entry_modules_release_context_only_clients(self):
        for module_name in D2_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                database = ContextOnlyDatabase()

                module._close_database(database)

                self.assertEqual(database.events, ["exit"])

    def test_cli_override_is_passed_to_client_as_an_absolute_path(self):
        module = importlib.import_module("D2.d2_2_vector_search")
        received_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "custom-seekdb"

            def client_factory(**kwargs):
                received_paths.append(kwargs["path"])
                return ExistingDatabase()

            with redirect_stdout(io.StringIO()):
                status = module.main(
                    client_factory=client_factory,
                    db_path=db_path,
                )

        self.assertEqual(status, 0)
        self.assertEqual(received_paths, [str(db_path.resolve())])

    def test_empty_results_do_not_print_false_success_claims(self):
        for module_name in (
            "D2.d2_2_vector_search",
            "D2.d2_3_hybrid_search",
            "D2.d2_4_compare",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                output = io.StringIO()

                with redirect_stdout(output):
                    module.run_demo(EmptyCollection())

                self.assertNotIn("✅", output.getvalue())


class D2ComparisonTests(unittest.TestCase):
    def test_natural_language_query_and_fulltext_keyword_are_separate(self):
        module = importlib.import_module("D2.d2_4_compare")
        collection = ComparisonCollection()

        with redirect_stdout(io.StringIO()):
            module.run_comparison(
                collection,
                query_text="错误码 E-4012 的解决方案",
                keyword="E-4012",
            )

        hybrid_call = collection.hybrid_calls[0]
        self.assertEqual(
            hybrid_call["query"]["where_document"],
            {"$contains": "E-4012"},
        )
        self.assertEqual(
            hybrid_call["knn"]["query_texts"],
            ["错误码 E-4012 的解决方案"],
        )


class D2ChunkingLifecycleTests(unittest.TestCase):
    def test_chunking_compare_accepts_cli_database_path_override(self):
        module = importlib.import_module("D2.d2_5_chunking_compare")
        expected = Path("/tmp/d2-test-seekdb")

        args = module.parse_args(["--db-path", str(expected)])

        self.assertEqual(args.db_path, expected)

    def test_seekdb_failure_degrades_to_memory_and_cleans_created_collection(self):
        module = importlib.import_module("D2.d2_5_chunking_compare")
        events = []

        class FailingCollection:
            def add(self, **_kwargs):
                events.append("add")
                raise RuntimeError("写入失败")

        class FailingDatabase:
            def has_collection(self, _name):
                return False

            def create_collection(self, name):
                events.append(("create", name))
                return FailingCollection()

            def delete_collection(self, name):
                events.append(("delete", name))

            def __exit__(self, *_args):
                events.append("exit")

        with redirect_stdout(io.StringIO()):
            row, backend = module.evaluate_with_fallback(
                "fixed",
                client_factory=lambda **_kwargs: FailingDatabase(),
            )

        self.assertEqual(backend, "memory")
        self.assertEqual(row["strategy"], "fixed")
        self.assertEqual(events[-2:], [("delete", "d2_chunking_fixed"), "exit"])


if __name__ == "__main__":
    unittest.main()
