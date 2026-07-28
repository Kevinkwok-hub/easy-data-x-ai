from __future__ import annotations

import subprocess
import sys
import unittest


class DependencyCompatibilityTest(unittest.TestCase):
    def test_fastapi_testclient_imports_without_deprecation_warning(self) -> None:
        """新版 Starlette 应直接使用 httpx2，而不是回退到弃用的 httpx。"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import warnings\n"
                    "from starlette.exceptions import StarletteDeprecationWarning\n"
                    "warnings.simplefilter('error', StarletteDeprecationWarning)\n"
                    "from fastapi.testclient import TestClient\n"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
