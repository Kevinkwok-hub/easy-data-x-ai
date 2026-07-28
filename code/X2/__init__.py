"""X2 示例包。

保留原有脚本式导入兼容性，同时允许通过 ``import X2...`` 使用。
"""

from __future__ import annotations

import sys
from pathlib import Path


_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
