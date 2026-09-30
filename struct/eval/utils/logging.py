"""CLI / 库共用的 loguru 初始化（各模块直接使用 ``from loguru import logger``）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _entry_script_label() -> str:
    """当前进程的入口脚本展示名（尽量区分 python -m 与直接 python xxx.py）。"""
    if not sys.argv:
        return ""
    p = Path(sys.argv[0])
    if p.name.lower() in ("python", "python.exe", "pythonw.exe"):
        mf = getattr(sys.modules.get("__main__"), "__file__", None)
        if mf:
            return Path(mf).name
        return p.name
    return p.name


def _log_record_patcher(record: Any) -> None:
    record["extra"]["entry_script"] = _entry_script_label()


def setup_cli_logging(verbose: int = 0) -> None:
    """
    将 loguru 默认 sink 重置为 stderr，格式与 CLI -v/-vv 一致。

    每条日志包含：入口脚本名、**调用处** 源码文件与行号、模块名、消息。

    - verbose==0：仅 WARNING 及以上
    - verbose>=1：INFO
    - verbose>=2：DEBUG
    """
    logger.remove()

    logger.configure(patcher=_log_record_patcher, extra={"entry_script": ""})

    lvl = "DEBUG" if verbose >= 2 else "INFO" if verbose >= 1 else "WARNING"
    logger.add(
        sys.stderr,
        level=lvl,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<magenta>{file.name}</magenta>:<magenta>{line}</magenta> | "
            "{message}"
        ),
        colorize=sys.stderr.isatty(),
    )
