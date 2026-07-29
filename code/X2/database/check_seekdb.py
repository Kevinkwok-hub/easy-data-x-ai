#!/usr/bin/env python3
"""
运行 X2 示例前检查 seekdb 连接。

Usage:
    python database/check_seekdb.py
    python database/check_seekdb.py --verbose
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.schema import DEFAULT_SEEKDB_PATH
from database.seekdb_client import (
    check_connection,
    resolve_mode,
    server_endpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 X2 使用的 seekdb 连接")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_SEEKDB_PATH,
        help=f"Embedded 数据路径（默认：{DEFAULT_SEEKDB_PATH}）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印解析后的连接配置",
    )
    args = parser.parse_args()

    mode = resolve_mode()
    print(f"模式: {mode}")
    if args.verbose and mode == "server":
        host, port = server_endpoint()
        print(f"  host: {host}")
        print(f"  port: {port}")

    ok, message = check_connection(args.db_path)
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
