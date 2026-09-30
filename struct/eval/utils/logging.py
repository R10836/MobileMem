"""Configure the shared Loguru logger used by the CLI and library."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _entry_script_label() -> str:
    """Return a useful entry-script label for module and direct execution."""
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
    """Reset Loguru to stderr using the CLI's ``-v``/``-vv`` format.

    Each record includes the entry script, call-site file and line, and
    message. Level zero shows warnings, one shows info, and two shows debug.
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
