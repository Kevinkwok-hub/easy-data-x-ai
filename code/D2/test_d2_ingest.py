import unittest

from D2.d2_1_ingest import chunk_document

class FakeCollection:
    def add(self, **_kwargs):
        return None

    def count(self):
        return 0


class FakeDatabase:
    def has_collection(self, _name):
        return False

    def create_collection(self, name):
        return FakeCollection()


class IngestChunkValidationTests(unittest.TestCase):
    def test_rejects_non_positive_size_and_invalid_overlap(self):
        invalid_parameters = (
            {"chunk_size": 0, "overlap": 0},
            {"chunk_size": -1, "overlap": 0},
            {"chunk_size": 3, "overlap": -1},
            {"chunk_size": 3, "overlap": 3},
            {"chunk_size": 3, "overlap": 4},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    chunk_document("", **parameters)


if __name__ == "__main__":
    unittest.main()
