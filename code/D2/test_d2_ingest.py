import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


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
    @classmethod
    def setUpClass(cls):
        fake_pyseekdb = types.ModuleType("pyseekdb")
        fake_pyseekdb.Client = FakeDatabase
        script_path = Path(__file__).with_name("d2_1_ingest.py")

        with (
            patch.dict(sys.modules, {"pyseekdb": fake_pyseekdb}),
            redirect_stdout(io.StringIO()),
        ):
            namespace = runpy.run_path(script_path)

        cls.chunk_document = staticmethod(namespace["chunk_document"])

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
                    self.chunk_document("", **parameters)


if __name__ == "__main__":
    unittest.main()
