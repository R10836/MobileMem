"""OpenAI-compatible Chat Completions HTTP client."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx
from loguru import logger

from eval import config


def chat_completion_for_model(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    response_format_json: bool | None = None,
) -> str:
    if not config.LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is not configured")
    if not config.LLM_API_KEY or config.LLM_API_KEY == "YOUR_API_KEY":
        raise RuntimeError(
            "LLM_API_KEY is missing or still set to YOUR_API_KEY"
        )

    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format_json is None:
        response_format_json = os.environ.get("LLM_RESPONSE_JSON", "1") == "1"
    if response_format_json:
        body["response_format"] = {"type": "json_object"}

    max_retries = config.llm_max_retries()
    retry_delay = config.llm_retry_delay_seconds()
    read_timeout = config.llm_read_timeout_seconds()
    timeout = httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)

    for attempt in range(1 + max_retries):
        if attempt > 0:
            time.sleep(retry_delay)
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM response has no choices: {data!r}")
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if not content:
                raise RuntimeError(f"LLM response has no content: {data!r}")
            return content if isinstance(content, str) else str(content)
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            n = attempt + 1
            total = 1 + max_retries
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                detail = " HTTP {} body_prefix={!r}".format(
                    e.response.status_code,
                    (e.response.text or "")[:300],
                )
            else:
                detail = ""
            if attempt >= max_retries:
                logger.error(
                    "LLM request reached the retry limit: attempt {}/{} model={} error={!r}{}",
                    n,
                    total,
                    model,
                    e,
                    detail,
                )
                raise
            logger.warning(
                "LLM request failed: attempt {}/{} model={} error={!r}{}; retrying in {}s",
                n,
                total,
                model,
                e,
                detail,
                retry_delay,
            )


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    response_format_json: bool | None = None,
) -> str:
    """Call the default LLM_MODEL for backward compatibility."""
    return chat_completion_for_model(
        messages,
        model=config.LLM_MODEL,
        temperature=temperature,
        response_format_json=response_format_json,
    )


def _safe_eval_integer_arithmetic(expr: str) -> int | None:
    """
    Evaluate an arithmetic expression emitted by the LLM, such as
    ``total_score_0_100: 40 + 10 - 10``. Only numbers, operators,
    parentheses, and whitespace are accepted to prevent code injection.
    """
    s = expr.strip().rstrip(",")
    if not s or s.startswith('"'):
        return None
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", s):
        return None
    try:
        v = eval(s, {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        return int(round(v))
    return None


def _fix_total_score_arithmetic_in_json_text(text: str) -> str:
    """Replace an arithmetic total_score expression with a JSON number."""

    def repl(m: re.Match[str]) -> str:
        prefix, raw = m.group(1), m.group(2).strip()
        n = _safe_eval_integer_arithmetic(raw)
        if n is not None:
            return f"{prefix}{n}"
        return m.group(0)

    return re.sub(
        r'("total_score_0_100"\s*:\s*)([^,\n}\]]+)',
        repl,
        text,
    )


def _repair_json_text_loose(text: str) -> str:
    """Remove trailing commas and extract the first complete JSON object."""
    s = (text or "").strip()
    if not s:
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    text = _fix_total_score_arithmetic_in_json_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json_text_loose(text))
