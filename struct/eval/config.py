"""Runtime configuration loaded from ``struct/.env`` and the environment."""

import os
import re
from pathlib import Path


def _load_dotenv() -> None:
    """Load a small, dependency-free subset of dotenv syntax."""
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

# OpenAI-compatible endpoint.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")


def judge_model_names() -> list[str]:
    """Return the ordered, deduplicated list of Judge model names.

    ``LLM_JUDGE_MODELS`` accepts comma, semicolon, and full-width comma
    separators. When it is unset, ``LLM_MODEL`` supplies the single model.
    """
    raw = os.environ.get("LLM_JUDGE_MODELS", "").strip()
    if raw:
        parts = [p.strip() for p in re.split(r"[,;，]", raw) if p.strip()]
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out if out else [LLM_MODEL.strip() or "gpt-4.1-mini"]
    m = LLM_MODEL.strip()
    return [m] if m else ["gpt-4.1-mini"]


def llm_config_ready() -> bool:
    return bool(LLM_API_KEY and LLM_API_KEY != "YOUR_API_KEY" and LLM_BASE_URL)


def judge_parallel_max_workers() -> int:
    """Return the LLM HTTP worker limit, configurable from 1 through 32."""
    raw = os.environ.get("LLM_JUDGE_MAX_WORKERS", "4").strip() or "4"
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 4


def tool_use_effectiveness_temperature() -> float:
    """Return the tool-effectiveness annotation temperature."""
    raw = os.environ.get("TOOL_USE_EFFECTIVENESS_TEMPERATURE", "0").strip() or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


def judge_temperature() -> float:
    """Return the Judge temperature; zero reduces repeat-evaluation variance."""
    raw = os.environ.get("LLM_JUDGE_TEMPERATURE", "0").strip() or "0"
    try:
        return max(0.0, min(2.0, float(raw)))
    except ValueError:
        return 0.0


def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default)).strip() or str(default)
    try:
        return max(lo, min(hi, int(raw)))
    except ValueError:
        return default


def _float_env(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name, str(default)).strip() or str(default)
    try:
        return max(lo, min(hi, float(raw)))
    except ValueError:
        return default


def llm_read_timeout_seconds() -> float:
    """Return the per-request Judge read timeout in seconds."""
    return _float_env("LLM_READ_TIMEOUT_SECONDS", 300.0, lo=10.0, hi=1800.0)


def llm_max_retries() -> int:
    """Return the maximum retries after the initial Judge request."""
    return _int_env("LLM_MAX_RETRIES", 3, lo=0, hi=20)


def llm_retry_delay_seconds() -> float:
    """Return the fixed delay between Judge requests in seconds."""
    return _float_env("LLM_RETRY_DELAY_SECONDS", 10.0, lo=0.0, hi=300.0)


def judge_final_reply_max_chars() -> int:
    """Return the final-reply character limit in the Judge payload."""
    return _int_env("JUDGE_FINAL_REPLY_MAX_CHARS", 48000, lo=4000, hi=200000)


def judge_thinking_max_chars() -> int:
    """Return the reasoning character limit in the Judge payload."""
    return _int_env("JUDGE_THINKING_MAX_CHARS", 24000, lo=2000, hi=200000)


def judge_tool_summary_max_chars() -> int:
    """Return the tool-summary character limit in the Judge payload."""
    return _int_env("JUDGE_TOOL_SUMMARY_MAX_CHARS", 16000, lo=2000, hi=200000)
