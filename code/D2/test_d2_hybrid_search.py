import io
import unittest
from contextlib import redirect_stdout

from D2.d2_3_hybrid_search import run_demo


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


class HybridSearchExampleTests(unittest.TestCase):
    def test_filtered_hybrid_search_filters_both_search_branches(self):
        collection = FakeCollection()

        with redirect_stdout(io.StringIO()):
            run_demo(collection)

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
