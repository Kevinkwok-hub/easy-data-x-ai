from __future__ import annotations

import importlib
import unittest


class McpSkillServerTests(unittest.TestCase):
    def test_import_does_not_start_server_and_tool_can_be_called_directly(self) -> None:
        """防止 CI 全绿但干净环境无法导入或调用 MCP 工具。"""
        module = importlib.import_module("mcp_skill_server")

        result = module.review_code_diff("+print('hello')", focus="correctness")

        self.assertIn("correctness", result)
        self.assertIn("15 characters", result)
        self.assertIn("Critical", result)


if __name__ == "__main__":
    unittest.main()
