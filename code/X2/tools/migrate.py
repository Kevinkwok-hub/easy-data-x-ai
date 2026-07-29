#!/usr/bin/env python3
"""
将 SKILL.md 迁移到 seekdb 的命令行工具。

Usage:
    python migrate.py <skill_file> [--db-path <path>] [--force]
    python migrate.py <skills_dir> --all [--db-path <path>] [--force]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services import MigrationService
from storage import create_storage
from database.schema import DEFAULT_SEEKDB_PATH
from database.seekdb_client import check_connection, ensure_database


def ensure_storage(db_path: str):
    try:
        ensure_database(path=db_path)
    except (ConnectionError, ValueError) as exc:
        print(exc)
        sys.exit(1)
    ok, message = check_connection(db_path)
    if not ok:
        print(message)
        sys.exit(1)
    storage = create_storage(db_path)
    if not storage.is_initialized():
        print(f"seekdb 尚未初始化：{db_path}")
        print("正在初始化集合...")
        storage.init(force=False)
    return storage


def migrate_file(skill_file: str, db_path: str, force: bool = False):
    skill_path = Path(skill_file)
    if not skill_path.exists():
        print(f"✗ 错误：文件不存在：{skill_file}")
        sys.exit(1)

    if skill_path.name != "SKILL.md":
        print(f"✗ 错误：文件名必须是 SKILL.md：{skill_file}")
        sys.exit(1)

    storage = ensure_storage(db_path)
    try:
        print(f"正在迁移：{skill_file}")
        result = MigrationService(storage).migrate_skill_file(
            skill_file,
            force=force,
        )
    finally:
        storage.close()

    if result["status"] == "success":
        print(f"✓ 成功：{result['skill_name']}")
        print(f"  规则：{result['rule_count']}")
        print(f"  示例：{result['example_count']}")
        print(f"  操作：{result['action']}")
    elif result["status"] == "skipped":
        print(f"⚠ 已跳过：{result['message']}")
    else:
        print(f"✗ 错误：{result.get('error', '未知错误')}")
        sys.exit(1)


def migrate_directory(skills_dir: str, db_path: str, force: bool = False):
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        print(f"✗ 错误：目录不存在：{skills_dir}")
        sys.exit(1)

    storage = ensure_storage(db_path)
    try:
        migration_service = MigrationService(storage)
        print(f"正在迁移目录中的全部 Skill：{skills_dir}")
        print("-" * 60)
        results = migration_service.migrate_directory(skills_dir, force=force)
        summary = migration_service.get_migration_summary()
    finally:
        storage.close()

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    print("-" * 60)
    print("迁移汇总：")
    print(f"  成功：{success_count}")
    print(f"  跳过：{skipped_count}")
    print(f"  失败：{error_count}")

    if success_count > 0:
        print("\n成功迁移：")
        for result in results:
            if result["status"] == "success":
                print(f"  ✓ {result['skill_name']}：{result['rule_count']} 条规则，"
                      f"{result['example_count']} 个示例")

    if error_count > 0:
        print("\n失败详情：")
        for result in results:
            if result["status"] == "error":
                print(f"  ✗ {result.get('skill_file', '未知文件')}：{result.get('error', '未知错误')}")
        sys.exit(1)

    print("\n数据库汇总：")
    print(f"  Skill 总数：{summary['skill_count']}")
    print(f"  规则总数：{summary['rule_count']}")
    print(f"  示例总数：{summary['example_count']}")
    if summary["category_counts"]:
        print(f"  分类统计：{summary['category_counts']}")


def main():
    parser = argparse.ArgumentParser(
        description="将 SKILL.md 文件迁移到 seekdb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python migrate.py skills/api-doc-writing/SKILL.md
  python migrate.py skills/ --all
  python migrate.py skills/ --all --force
        """,
    )
    parser.add_argument("path", help="SKILL.md 文件或 Skill 目录路径")
    parser.add_argument("--all", action="store_true", help="迁移目录中的全部 SKILL.md")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_SEEKDB_PATH,
        help=f"seekdb 路径（默认：{DEFAULT_SEEKDB_PATH}）",
    )
    parser.add_argument("--force", action="store_true", help="强制更新已存在的 Skill")
    args = parser.parse_args()

    path = Path(args.path)
    if args.all:
        if not path.is_dir():
            print(f"✗ 错误：使用 --all 时必须提供目录：{args.path}")
            sys.exit(1)
        migrate_directory(str(path), args.db_path, force=args.force)
    else:
        if path.is_dir():
            print(f"✗ 错误：当前路径是目录，请添加 --all：{args.path}")
            sys.exit(1)
        migrate_file(str(path), args.db_path, force=args.force)


if __name__ == "__main__":
    main()
