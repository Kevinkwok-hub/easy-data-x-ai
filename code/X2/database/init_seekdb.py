#!/usr/bin/env python3
"""
初始化 X2 Skill 存储所需的 seekdb 集合。

Usage:
    python init_seekdb.py [--db-path <path>] [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage import create_storage
from database.schema import DEFAULT_SEEKDB_PATH
from database.seekdb_client import check_connection, ensure_database


def main():
    parser = argparse.ArgumentParser(description="初始化 X2 seekdb Skill 存储")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_SEEKDB_PATH,
        help=f"seekdb 路径（默认：{DEFAULT_SEEKDB_PATH}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新创建已存在的集合",
    )
    args = parser.parse_args()

    try:
        ensure_database(path=args.db_path)
    except (ConnectionError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    ok, message = check_connection(args.db_path)
    if not ok:
        print(message)
        sys.exit(1)

    storage = create_storage(args.db_path)
    try:
        storage.init(force=args.force)
        print(f"✓ seekdb 已就绪：{args.db_path}")
        summary = storage.get_migration_summary()
        print(f"  Skill：{summary['skill_count']}，规则：{summary['rule_count']}，"
              f"示例：{summary['example_count']}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
