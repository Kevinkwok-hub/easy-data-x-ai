from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


X1_ROOT = Path(__file__).resolve().parent


class X1EntrypointTests(unittest.TestCase):
    def test_all_fifteen_examples_run_from_unrelated_working_directory(self) -> None:
        """防止示例偷偷依赖仓库当前目录或遗漏入口。"""
        scripts = sorted(X1_ROOT.rglob("x1_*.py"))
        self.assertEqual(len(scripts), 15)

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUTF8"] = "1"
        with tempfile.TemporaryDirectory(prefix="easy-data-x-ai-x1-") as temp_dir:
            for script in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, str(script)],
                        cwd=temp_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=f"{script.name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                    )
                    self.assertTrue(result.stdout.strip(), msg=f"{script.name} 没有产生演示输出")
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
