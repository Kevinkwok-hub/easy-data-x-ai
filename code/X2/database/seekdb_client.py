"""Unified seekdb connection helpers for X2."""

from __future__ import annotations

import os
import platform
import socket
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pyseekdb
from dotenv import load_dotenv

from database.schema import DEFAULT_SEEKDB_PATH

X2_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_PYLIBSEEKDB_AVAILABLE = find_spec("pylibseekdb") is not None
DEFAULT_DATABASE = "x2_skills"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2881
DEFAULT_TENANT = "sys"
DEFAULT_USER = "root"

SeekdbMode = Literal["embedded", "server"]


def load_x2_env(path: str | Path = X2_ENV_PATH) -> None:
    # 配置固定从 X2/.env 读取，已有系统环境变量优先。
    load_dotenv(dotenv_path=path, override=False)


def resolve_mode() -> SeekdbMode:
    """Pick embedded vs server mode from env and platform capabilities."""
    explicit = os.environ.get("SEEKDB_MODE", "").strip().lower()
    if explicit in ("server", "remote"):
        return "server"
    if explicit in ("embedded", "local"):
        return "embedded"
    if os.environ.get("SEEKDB_HOST"):
        return "server"
    if not _PYLIBSEEKDB_AVAILABLE:
        return "server"
    return "embedded"


def server_endpoint() -> tuple[str, int]:
    host = os.environ.get("SEEKDB_HOST", DEFAULT_HOST)
    raw_port = os.environ.get("SEEKDB_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("SEEKDB_PORT 必须是 1 到 65535 之间的整数") from exc
    if not 1 <= port <= 65535:
        raise ValueError("SEEKDB_PORT 必须是 1 到 65535 之间的整数")
    return host, port


def create_client(path: str = DEFAULT_SEEKDB_PATH) -> pyseekdb.Client:
    """Create a pyseekdb client using the resolved deployment mode."""
    load_x2_env()
    mode = resolve_mode()
    database = os.environ.get("SEEKDB_DATABASE", DEFAULT_DATABASE)
    password = os.environ.get("SEEKDB_PASSWORD", "")
    user = os.environ.get("SEEKDB_USER", DEFAULT_USER)
    tenant = os.environ.get("SEEKDB_TENANT", DEFAULT_TENANT)

    if mode == "server":
        host, port = server_endpoint()
        return pyseekdb.Client(
            host=host,
            port=port,
            tenant=tenant,
            database=database,
            user=user,
            password=password,
        )
    return pyseekdb.Client(path=path, database=database)


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _server_unready_message(host: str, port: int) -> str:
    return f"seekdb Server 未就绪：{host}:{port}。请先执行 docker compose up -d。"


def connection_hint(mode: SeekdbMode | None = None) -> str:
    mode = mode or resolve_mode()
    lines = [
        "seekdb 未就绪。请按以下步骤启动：",
        "",
        "  cd code/X2",
        "  docker compose up -d",
        "  python database/check_seekdb.py",
        "",
    ]
    if mode == "server":
        host, port = server_endpoint()
        lines.extend([
            f"  当前配置: {host}:{port} (Server 模式)",
            "  可复制 .env.example 为 .env 后按需修改 SEEKDB_* 变量。",
        ])
    else:
        lines.extend([
            "  Linux 嵌入式模式可直接使用，无需 Docker。",
            f"  数据目录: {DEFAULT_SEEKDB_PATH}",
        ])
    if platform.system() == "Darwin":
        lines.append("")
        lines.append("  提示: macOS 不支持嵌入式 seekdb，课程推荐使用 Docker Server 模式。")
    return "\n".join(lines)


def check_connection(path: str = DEFAULT_SEEKDB_PATH) -> tuple[bool, str]:
    """Return (ok, message). Does not raise."""
    load_x2_env()
    mode = resolve_mode()
    database = os.environ.get("SEEKDB_DATABASE", DEFAULT_DATABASE)

    if mode == "server":
        try:
            host, port = server_endpoint()
        except ValueError as exc:
            return False, str(exc)
        if not _port_open(host, port):
            return False, _server_unready_message(host, port)

    try:
        with create_client(path) as client:
            client.list_collections()
        return True, f"✓ seekdb 可用 ({mode} 模式, database={database})"
    except Exception as exc:
        return False, f"连接失败: {exc}\n\n{connection_hint(mode)}"


def require_connection(path: str = DEFAULT_SEEKDB_PATH) -> pyseekdb.Client:
    ok, message = check_connection(path)
    if not ok:
        raise ConnectionError(message)
    return create_client(path)


def ensure_database(
    name: str | None = None,
    path: str = DEFAULT_SEEKDB_PATH,
) -> None:
    """Create the target database before binding a database client."""
    load_x2_env()
    db_name = name or os.environ.get("SEEKDB_DATABASE", DEFAULT_DATABASE)
    mode = resolve_mode()
    # 建库必须使用 AdminClient；普通 Client 不能绑定尚不存在的 database。
    if mode == "embedded":
        admin_kwargs = {"path": path}
    else:
        host, port = server_endpoint()
        if not _port_open(host, port):
            raise ConnectionError(_server_unready_message(host, port))
        tenant = os.environ.get("SEEKDB_TENANT", DEFAULT_TENANT)
        user = os.environ.get("SEEKDB_USER", DEFAULT_USER)
        password = os.environ.get("SEEKDB_PASSWORD", "")
        admin_kwargs = {
            "host": host,
            "port": port,
            "tenant": tenant,
            "user": user,
            "password": password,
        }
    with pyseekdb.AdminClient(**admin_kwargs) as admin:
        existing = {db.name for db in admin.list_databases()}
        if db_name not in existing:
            admin.create_database(db_name)
