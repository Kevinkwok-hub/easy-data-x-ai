#!/usr/bin/env python3
"""
从 seekdb 查询 Skill 的命令行工具。

Usage:
    python query_tool.py [--db-path <path>] get <skill_name> [--format <format>]
    python query_tool.py [--db-path <path>] list [--category <cat>]
    python query_tool.py [--db-path <path>] search <keyword>
    python query_tool.py [--db-path <path>] rules <skill_name>
    python query_tool.py [--db-path <path>] examples <skill_name>
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services import QueryService, SkillService
from storage import create_storage
from database.schema import DEFAULT_SEEKDB_PATH
from database.seekdb_client import check_connection


def ensure_storage(db_path: str):
    ok, message = check_connection(db_path)
    if not ok:
        print(message)
        sys.exit(1)
    storage = create_storage(db_path)
    if not storage.is_initialized():
        print(f"✗ 错误：seekdb 尚未初始化：{db_path}")
        print("  请先运行：python database/init_seekdb.py")
        print("  或运行：python tools/migrate.py skills/ --all")
        storage.close()
        sys.exit(1)
    return storage


def get_skill(skill_name: str, storage, output_format: str = "text"):
    query_service = QueryService(storage)
    skill_complete = query_service.get_skill_complete(skill_name)

    if not skill_complete:
        print(f"✗ 未找到 Skill：{skill_name}")
        sys.exit(1)

    if output_format == "json":
        print(json.dumps(skill_complete, indent=2, default=str))
    else:
        skill = skill_complete["skill"]
        print(f"Skill：{skill['name']}")
        print(f"描述：{skill['description']}")
        print(f"分类：{skill.get('category', '无')}")
        print(f"版本：{skill.get('version', '无')}")
        print(f"状态：{skill.get('status', '无')}")
        print(f"规则：{skill_complete['rule_count']}")
        print(f"示例：{skill_complete['example_count']}")
        print(f"\n内容长度：{len(skill['content'])} 个字符")


def list_skills(category: str = None, storage=None):
    skill_service = SkillService(storage)
    skills = skill_service.list_skills(category=category)

    if not skills:
        print("未找到 Skill。")
        if category:
            print(f"分类过滤条件：{category}")
        sys.exit(0)

    print(f"找到 {len(skills)} 个 Skill：")
    print("-" * 60)
    for skill in skills:
        print(f"  {skill.name}")
        print(f"    分类：{skill.category or '无'}")
        desc = skill.description[:60] + ("..." if len(skill.description) > 60 else "")
        print(f"    描述：{desc}")
        print()


def search_skills(keyword: str, storage):
    query_service = QueryService(storage)
    skills = query_service.search_skills(keyword)

    if not skills:
        print(f"没有匹配关键词的 Skill：{keyword}")
        sys.exit(0)

    print(f"找到 {len(skills)} 个匹配“{keyword}”的 Skill：")
    print("-" * 60)
    for skill in skills:
        print(f"  {skill.name}")
        print(f"    分类：{skill.category or '无'}")
        desc = skill.description[:60] + ("..." if len(skill.description) > 60 else "")
        print(f"    描述：{desc}")
        print()


def get_rules(skill_name: str, storage):
    query_service = QueryService(storage)
    rules = query_service.get_rules_by_skill(skill_name)

    if not rules:
        print(f"该 Skill 没有规则：{skill_name}")
        sys.exit(0)

    print(f"“{skill_name}”的规则：{len(rules)} 条")
    print("-" * 60)
    for i, rule in enumerate(rules, 1):
        print(f"{i}. [{rule.rule_type}] {rule.rule_key}")
        print(f"   内容：{rule.rule_value[:80]}...")
        if rule.rule_description:
            print(f"   描述：{rule.rule_description[:60]}...")
        print(f"   优先级：{rule.priority}")
        print()


def get_examples(skill_name: str, storage):
    query_service = QueryService(storage)
    examples = query_service.get_examples_by_skill(skill_name)

    if not examples:
        print(f"该 Skill 没有示例：{skill_name}")
        sys.exit(0)

    print(f"“{skill_name}”的示例：{len(examples)} 个")
    print("-" * 60)
    for i, example in enumerate(examples, 1):
        print(f"{i}. 类型：{example.example_type or 'text'}")
        if example.title:
            print(f"   标题：{example.title}")
        print(f"   代码：{example.code[:80]}...")
        if example.result:
            print(f"   结果：{example.result[:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="从 seekdb 查询 Skill")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_SEEKDB_PATH,
        help=f"seekdb 路径（默认：{DEFAULT_SEEKDB_PATH}）",
    )

    subparsers = parser.add_subparsers(dest="command", help="要执行的命令")

    get_parser = subparsers.add_parser("get", help="查看 Skill 信息")
    get_parser.add_argument("skill_name", help="Skill 名称")
    get_parser.add_argument("--format", choices=["text", "json"], default="text")

    list_parser = subparsers.add_parser("list", help="列出全部 Skill")
    list_parser.add_argument("--category", help="按分类过滤")

    search_parser = subparsers.add_parser("search", help="混合检索 Skill")
    search_parser.add_argument("keyword", help="检索关键词")

    rules_parser = subparsers.add_parser("rules", help="查看 Skill 规则")
    rules_parser.add_argument("skill_name", help="Skill 名称")

    examples_parser = subparsers.add_parser("examples", help="查看 Skill 示例")
    examples_parser.add_argument("skill_name", help="Skill 名称")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    storage = ensure_storage(args.db_path)
    try:
        if args.command == "get":
            get_skill(args.skill_name, storage, args.format)
        elif args.command == "list":
            list_skills(category=args.category, storage=storage)
        elif args.command == "search":
            search_skills(args.keyword, storage)
        elif args.command == "rules":
            get_rules(args.skill_name, storage)
        elif args.command == "examples":
            get_examples(args.skill_name, storage)
    finally:
        storage.close()


if __name__ == "__main__":
    main()
