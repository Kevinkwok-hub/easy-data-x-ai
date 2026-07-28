import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class FakeCollection:
    def __init__(self):
        self.hybrid_calls = []

    def count(self):
        return 0

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def has_collection(self, _name):
        return True

    def get_collection(self, _name):
        return self.collection


class HybridSearchExampleTests(unittest.TestCase):
    def test_filtered_hybrid_search_filters_both_search_branches(self):
        collection = FakeCollection()
        fake_pyseekdb = types.ModuleType("pyseekdb")
        fake_pyseekdb.Client = lambda: FakeDatabase(collection)
        script_path = Path(__file__).with_name("d2_3_hybrid_search.py")

        with (
            patch.dict(sys.modules, {"pyseekdb": fake_pyseekdb}),
            redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(script_path)

        filtered_call = collection.hybrid_calls[2]
        self.assertEqual(
            filtered_call,
            {
                "query": {
                    "where_document": {"$contains": "性能优化"},
                    "where": {"version": "4.2"},
                    "n_results": 5,
                },
                "knn": {
                    "query_texts": ["性能优化"],
                    "where": {"version": "4.2"},
                    "n_results": 5,
                },
                "rank": {"rrf": {}},
                "n_results": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
