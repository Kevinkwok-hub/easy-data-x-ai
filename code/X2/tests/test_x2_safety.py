from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml


X2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X2_ROOT))

from database import init_seekdb
from parsers.markdown_parser import MarkdownParser
from parsers.rule_extractor import RuleExtractor
from tools import migrate


class MarkdownParserSafetyTests(unittest.TestCase):
    def _write_skill(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        return path

    def test_rejects_non_mapping_frontmatter(self) -> None:
        path = self._write_skill("---\n- invalid\n---\n正文")

        with self.assertRaisesRegex(ValueError, "frontmatter"):
            MarkdownParser(str(path)).parse()

    def test_rejects_missing_or_non_string_name(self) -> None:
        for frontmatter in ({}, {"name": 123}):
            with self.subTest(frontmatter=frontmatter):
                path = self._write_skill(
                    f"---\n{yaml.safe_dump(frontmatter, allow_unicode=True)}---\n正文"
                )
                with self.assertRaisesRegex(ValueError, "name"):
                    MarkdownParser(str(path)).parse()


class RuleKeySafetyTests(unittest.TestCase):
    def test_chinese_rules_receive_unique_stable_keys(self) -> None:
        content = """## Formatting rules
- 必须使用明确的标题
- 必须保留原始示例
"""
        first = RuleExtractor(content).extract("demo")
        second = RuleExtractor(content).extract("demo")

        self.assertEqual(len(first), 2)
        self.assertEqual([rule.rule_key for rule in first], [rule.rule_key for rule in second])
        self.assertEqual(len({rule.rule_key for rule in first}), 2)
        self.assertTrue(all(key.startswith("format_rule_") for key in (rule.rule_key for rule in first)))

    def test_mixed_chinese_rules_with_same_ascii_slug_are_unique_and_stable(self) -> None:
        content = """## Formatting rules
- API 必须使用认证保护
- API 必须限制请求频率
"""
        first = RuleExtractor(content).extract("demo")
        second = RuleExtractor(content).extract("demo")

        first_keys = [rule.rule_key for rule in first]
        self.assertEqual(first_keys, [rule.rule_key for rule in second])
        self.assertEqual(len(first_keys), 2)
        self.assertEqual(len(set(first_keys)), 2)
        self.assertTrue(all(key.startswith("api_") for key in first_keys))


class DatabaseInitializationTests(unittest.TestCase):
    def test_server_database_is_ensured_before_connection_check(self) -> None:
        calls: list[str] = []
        storage = Mock()
        storage.get_migration_summary.return_value = {
            "skill_count": 0,
            "rule_count": 0,
            "example_count": 0,
        }

        with (
            patch.object(
                init_seekdb,
                "ensure_database",
                side_effect=lambda **_kwargs: calls.append("ensure"),
            ),
            patch.object(
                init_seekdb,
                "check_connection",
                side_effect=lambda _path: (calls.append("check") or (True, "ok")),
            ),
            patch.object(init_seekdb, "create_storage", return_value=storage),
            patch.object(sys, "argv", ["init_seekdb.py"]),
        ):
            init_seekdb.main()

        self.assertEqual(calls, ["ensure", "check"])

    def test_migration_ensures_database_before_connection_check(self) -> None:
        calls: list[str] = []
        storage = Mock()
        storage.is_initialized.return_value = True

        with (
            patch.object(
                migrate,
                "ensure_database",
                side_effect=lambda **_kwargs: calls.append("ensure"),
            ),
            patch.object(
                migrate,
                "check_connection",
                side_effect=lambda _path: (calls.append("check") or (True, "ok")),
            ),
            patch.object(migrate, "create_storage", return_value=storage),
        ):
            migrate.ensure_storage("server-db")

        self.assertEqual(calls, ["ensure", "check"])


class DockerComposeSafetyTests(unittest.TestCase):
    def test_seekdb_ports_bind_to_loopback(self) -> None:
        compose = yaml.safe_load((X2_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        self.assertEqual(
            compose["services"]["seekdb"]["ports"],
            ["127.0.0.1:2881:2881", "127.0.0.1:2886:2886"],
        )


if __name__ == "__main__":
    unittest.main()
