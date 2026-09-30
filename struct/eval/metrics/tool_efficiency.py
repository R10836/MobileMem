"""工具调用效率：T、R、Q 与 Score = Q × R × 1/(ln(1+T)+1)。"""

from __future__ import annotations

import json
import math
from typing import Any

from eval.traces.jsonl import ConversationTurn


def _args_key(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def _parse_tool_result_payload(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _is_empty_result(raw: str, is_error: bool) -> bool:
    if is_error:
        return False
    if not raw.strip():
        return True
    p = _parse_tool_result_payload(raw)
    if p is None:
        return len(raw.strip()) < 4
    if isinstance(p, dict):
        if p.get("status") == "error" or "error" in p:
            return False
        r = p.get("results")
        if isinstance(r, list) and len(r) == 0:
            return True
        if p == {}:
            return True
    if isinstance(p, list) and len(p) == 0:
        return True
    return False


def _is_error_result(is_error: bool, raw: str, details: dict[str, Any] | None) -> bool:
    if is_error:
        return True
    if details and str(details.get("status", "")).lower() == "error":
        return True
    p = _parse_tool_result_payload(raw)
    if isinstance(p, dict):
        if p.get("status") == "error" or p.get("error"):
            return True
    return False


def compute_tool_efficiency(turn: ConversationTurn, quality_q: float) -> dict[str, Any]:
    """
    quality_q: 任务完成质量 Q∈[0,1]，通常来自 Judge 总分/100。
    工具调用与 ``toolResult`` 按 ``toolCallId`` 配对（见 ``ConversationTurn.iter_tool_call_events``），
    不再使用 ``calls[i]`` 与 ``tool_results[i]`` 下标。
    """
    events = turn.iter_tool_call_events()
    T = len(events)
    if T == 0:
        return {
            "T": 0,
            "R": 1.0,
            "Q": max(0.0, min(1.0, quality_q)),
            "score": max(0.0, min(1.0, quality_q)),
            "invalid_count": 0,
            "invalid_reasons": [],
        }

    invalid = 0
    reasons: list[str] = []

    prev_key: str | None = None
    for ev in events:
        key = f"{ev.name}\0{_args_key(ev.arguments)}"
        bad = False
        if prev_key is not None and key == prev_key:
            invalid += 1
            reasons.append(f"连续相同调用: {ev.name}")
            bad = True
        prev_key = key

        res = ev.result
        if res is None:
            if not bad:
                invalid += 1
                reasons.append(f"无配对 toolResult: {ev.name} id={ev.tool_call_id!r}")
            continue
        if bad:
            continue
        if _is_error_result(res.is_error, res.raw_content, res.details):
            invalid += 1
            reasons.append(f"报错调用: {ev.name}")
        elif _is_empty_result(res.raw_content, res.is_error):
            invalid += 1
            reasons.append(f"空结果: {ev.name}")

    R = (T - invalid) / T if T else 1.0
    R = max(0.0, min(1.0, R))
    Q = max(0.0, min(1.0, quality_q))
    denom = math.log(1 + T) + 1
    score = Q * R * (1.0 / denom)

    return {
        "T": T,
        "R": round(R, 6),
        "Q": Q,
        "score": round(score, 6),
        "invalid_count": invalid,
        "invalid_reasons": reasons,
    }
