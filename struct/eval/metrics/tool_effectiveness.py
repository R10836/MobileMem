"""Measure effective tool-call ratio from paired JSONL traces and LLM labels.

Potential P2 work includes concurrent chunks and richer structured summaries;
the current implementation sends one batched request.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Optional

from loguru import logger

from eval import config
from eval.judging.client import chat_completion_for_model, parse_json_loose
from eval.traces.jsonl import ConversationTurn, ToolCallEvent
from eval.core.tasks import TaskItem
from eval.metrics.tool_efficiency import _args_key, _is_error_result, _parse_tool_result_payload

_ANNOTATOR_REF_PREFIX = "tue_"
_FINAL_ASSISTANT_MAX = 8000
# Total annotation budget for structured summaries and optional raw excerpts.
_DEFAULT_TOOL_RESULT_BUDGET = 2800
# Per-field scalar limit prevents one error string from using the full budget.
_DIGEST_SCALAR_STR_MAX = 400
# Head and tail shares when including raw non-JSON or summarized content.
_RAW_HEAD_FRAC = 0.45
_RAW_TAIL_FRAC = 0.35
# Mapping keys prioritized for status, error, and result-size summaries.
_DIGEST_PRIORITY_KEYS = (
    "ok",
    "success",
    "status",
    "error",
    "errors",
    "message",
    "code",
    "detail",
    "details",
    "results",
    "data",
    "items",
    "rows",
    "records",
    "count",
    "total",
    "nextPageToken",
)


def annotator_ref_for_event_index(i: int) -> str:
    return f"{_ANNOTATOR_REF_PREFIX}{i}"


def _tool_result_budget_chars() -> int:
    raw = os.environ.get("TOOL_USE_EFFECTIVENESS_RESULT_BUDGET", "").strip()
    if not raw:
        return _DEFAULT_TOOL_RESULT_BUDGET
    try:
        return max(800, min(16000, int(raw)))
    except ValueError:
        return _DEFAULT_TOOL_RESULT_BUDGET


def _truncate_str(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _digest_json_value(val: Any, *, depth: int, max_depth: int, budget: list[int]) -> str:
    """Recursively append compact text while decrementing a shared budget."""
    if budget[0] <= 0:
        return "…"
    if depth > max_depth:
        return type(val).__name__

    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        s = repr(val)
        budget[0] -= len(s)
        return s
    if isinstance(val, str):
        t = _truncate_str(val, min(_DIGEST_SCALAR_STR_MAX, max(20, budget[0])))
        budget[0] -= len(t)
        return json.dumps(t, ensure_ascii=False)

    if isinstance(val, list):
        n = len(val)
        budget[0] -= 20
        if n == 0:
            return "[]"
        head = min(3, n)
        parts = [f"array[len={n}]"]
        for i in range(head):
            if budget[0] <= 0:
                parts.append("…")
                break
            parts.append(f"[{i}]={_digest_json_value(val[i], depth=depth + 1, max_depth=max_depth, budget=budget)}")
        if n > head:
            parts.append(f"…(+{n - head})")
        return " ".join(parts)

    if isinstance(val, dict):
        keys = list(val.keys())
        priority = [k for k in _DIGEST_PRIORITY_KEYS if k in val]
        rest = [k for k in keys if k not in priority][: max(0, 12 - len(priority))]
        ordered = priority + rest
        parts = [f"dict[keys={len(keys)}]"]
        budget[0] -= 30
        for k in ordered:
            if budget[0] <= 0:
                parts.append("…")
                break
            sub = _digest_json_value(val[k], depth=depth + 1, max_depth=max_depth, budget=budget)
            parts.append(f"{k}={sub}")
        if len(ordered) < len(keys):
            parts.append(f"…(+{len(keys) - len(ordered)}keys)")
        return " ".join(parts)

    s = str(type(val).__name__)
    budget[0] -= len(s)
    return s


def _structured_digest_line(parsed: Any) -> str:
    b = [2000]
    if isinstance(parsed, dict):
        return _digest_json_value(parsed, depth=0, max_depth=4, budget=b)
    if isinstance(parsed, list):
        return _digest_json_value(parsed, depth=0, max_depth=4, budget=b)
    return str(parsed)[:500]


def _tool_result_text_digest(raw: str, *, char_budget: int) -> str:
    """
    Parse long JSON content into a compact summary and retain head and tail
    windows so errors near the end are not lost.
    """
    raw = (raw or "").strip()
    if not raw:
        return "（空）"

    structured = ""
    parsed = _parse_tool_result_payload(raw)
    if parsed is not None:
        structured = "[结构化] " + _structured_digest_line(parsed)

    reserve = len(structured) + 80 if structured else 40
    raw_budget = max(120, char_budget - reserve)
    if len(raw) <= raw_budget:
        tail = raw if not structured else f"{structured}\n[全文]\n{raw}"
        return tail

    head_n = max(80, int(raw_budget * _RAW_HEAD_FRAC))
    tail_n = max(80, int(raw_budget * _RAW_TAIL_FRAC))
    omitted = len(raw) - head_n - tail_n
    if omitted < 0:
        head_n = raw_budget // 2
        tail_n = raw_budget - head_n
        omitted = max(0, len(raw) - head_n - tail_n)
    head = raw[:head_n]
    tail = raw[-tail_n:] if tail_n else ""
    win = (
        f"[… 省略中间约 {omitted} 字符 …]\n"
        f"--- 原文首部 ({head_n}) ---\n{head}\n"
        f"--- 原文尾部 ({tail_n}) ---\n{tail}"
    )
    if structured:
        return structured + "\n" + win
    return win


def _result_summary(ev: ToolCallEvent) -> str:
    r = ev.result
    if r is None:
        return "（无 toolResult，toolCallId 未在日志中匹配到返回）"
    budget = _tool_result_budget_chars()
    body = _tool_result_text_digest(r.raw_content or "", char_budget=budget)
    return f"isError={r.is_error}\n{body}"


def _payload_top_level_ok_false(raw: str) -> bool:
    p = _parse_tool_result_payload(raw)
    if isinstance(p, dict) and p.get("ok") is False:
        return True
    return False


def _rule_prefilter_event(
    ev: ToolCallEvent,
    *,
    prev_call_key: str | None,
) -> tuple[str | None, str | None]:
    """
    ``(reason, None)`` means a rule marked the call ineffective;
    ``(None, None)`` delegates the decision to the LLM.
    """
    key = f"{ev.name}\0{_args_key(ev.arguments)}"
    if prev_call_key is not None and key == prev_call_key:
        return "rule:consecutive_duplicate", "连续相同 name+arguments 调用"

    r = ev.result
    if r is None:
        if ev.tool_call_id:
            return "rule:no_tool_result", "无配对 toolResult（按 toolCallId）"
        return "rule:no_tool_call_id", "toolCall 无 id 且无配对结果"

    if _is_error_result(r.is_error, r.raw_content, r.details):
        return "rule:tool_error", "工具返回报错（isError 或 payload 错误）"

    if _payload_top_level_ok_false(r.raw_content):
        return "rule:payload_ok_false", "返回 JSON 顶层 ok=false"

    return None, None


def _rule_dim_fields_false() -> dict[str, bool]:
    return {
        "task_relevance": False,
        "action_coherence": False,
        "result_usefulness": False,
    }


TOOL_USE_ANNOTATOR_SYSTEM = """你是手机 AI Agent 会话评测中的「工具调用有效性」标注裁判。

【字段说明】
- **task_relevance**：该次工具调用/工具选择是否与当前用户任务相关（结合 task_prompt、user_query、expected_behavior）。
- **action_coherence**：是否与当前 thinking（thinking_before_call、thinking_message_full）一致，推理与工具选择无明显矛盾。
- **result_usefulness**：工具返回对完成任务是否有信息量/可用。**禁止仅凭结果为空或极短判 false**；若结合 thinking 可判断为合理试探（如规范 COUNT、空集），可为 true。
- **effective**：沿用经典「工具调用是否有效」的综合判断——与任务相关且合理，或返回对完成任务有明显帮助；与三字段分别独立输出，**不**由三字段机械推导。

【输出】
只输出一个 JSON 对象（不要 Markdown 代码块），格式如下：
{"judgments":[{"annotator_ref":"与输入 tools_to_judge 完全一致","tool_call_id":"日志中的 id，可为空字符串","task_relevance":true或false,"action_coherence":true或false,"result_usefulness":true或false,"effective":true或false,"reason":"一句中文理由"}]}
必须覆盖输入中列出的**每一条** annotator_ref（顺序可任意）。"""


def _unique_tool_call_id_to_ref(
    pending_events: list[ToolCallEvent],
    pending_refs: list[str],
) -> dict[str, str]:
    """Use a unique nonempty tool_call_id when annotator_ref is missing."""
    if len(pending_events) != len(pending_refs):
        return {}
    counts: Counter[str] = Counter()
    first_ref_by_tid: dict[str, str] = {}
    for ev, ref in zip(pending_events, pending_refs):
        tid = ev.tool_call_id
        if not tid:
            continue
        s = str(tid)
        counts[s] += 1
        first_ref_by_tid.setdefault(s, ref)
    return {tid: r for tid, r in first_ref_by_tid.items() if counts[tid] == 1}


def _bool_from_item(item: dict[str, Any], key: str) -> bool:
    v = item.get(key)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "是")
    return bool(v)


def _build_annotator_user_payload(
    task: TaskItem,
    user_query: str,
    turn: ConversationTurn,
    pending_events: list[ToolCallEvent],
    pending_refs: list[str],
) -> str:
    blocks = []
    for ref, ev in zip(pending_refs, pending_events):
        tid = ev.tool_call_id if ev.tool_call_id is not None else ""
        blocks.append(
            {
                "annotator_ref": ref,
                "tool_call_id": tid,
                "name": ev.name,
                "arguments_json": json.dumps(ev.arguments, ensure_ascii=False)[:2000],
                "tool_result_summary": _result_summary(ev),
                "thinking_before_call": (ev.thinking_before_call or "")[:4000],
                "thinking_message_full": (ev.thinking_message_full or "")[:6000],
            }
        )
    fat = turn.final_assistant_text()
    if len(fat) > _FINAL_ASSISTANT_MAX:
        fat = fat[:_FINAL_ASSISTANT_MAX] + "…"
    payload = {
        "task_prompt": task.prompt[:8000],
        "expected_behavior": (task.expected_behavior or "")[:4000],
        "user_query": user_query[:4000],
        "final_assistant_text": fat,
        "tools_to_judge": blocks,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


JudgmentTuple = tuple[bool, bool, bool, bool, str]


def _parse_judgments(
    raw: str,
    expected_refs: set[str],
    id_fallback: dict[str, str],
) -> dict[str, JudgmentTuple]:
    """
    annotator_ref -> (task_relevance, action_coherence, result_usefulness, effective, reason)
    """
    data = parse_json_loose(raw)
    arr = data.get("judgments") or data.get("results") or []
    out: dict[str, JudgmentTuple] = {}
    if not isinstance(arr, list):
        return out
    for item in arr:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("annotator_ref") or item.get("annotatorRef") or "").strip()
        if not ref:
            tid = str(item.get("tool_call_id", "") or "")
            ref = id_fallback.get(tid, "")
        if ref not in expected_refs:
            continue
        tr = _bool_from_item(item, "task_relevance")
        ac = _bool_from_item(item, "action_coherence")
        ru = _bool_from_item(item, "result_usefulness")
        eff = _bool_from_item(item, "effective")
        reason = str(item.get("reason", "") or "").strip() or "（无理由）"
        out[ref] = (tr, ac, ru, eff, reason)
    return out


def _annotate_user_repair_suffix(expected_refs: set[str]) -> str:
    refs = ",".join(sorted(expected_refs))
    return (
        "\n\n【修复要求】上次输出无法解析为合法 JSON，或缺少部分 annotator_ref。"
        "请仅输出一个 JSON 对象，judgments 必须覆盖下列每一个 annotator_ref："
        f"{refs}"
    )


def compute_tool_use_effectiveness(turn: ConversationTurn, task: TaskItem) -> dict[str, Any]:
    """
    Apply deterministic rules before LLM annotation. When ``T=0``, return an
    effective ratio of 1.0 without calling the LLM.
    """
    events = turn.iter_tool_call_events()
    T = len(events)
    if T > 0:
        logger.debug(
            "Tool-effectiveness evaluation started task_id={} tool_call_events={}",
            task.task_id,
            T,
        )
    if T == 0:
        logger.debug(
            "Tool effectiveness task_id={} has no calls; "
            "T=0 effective_ratio=1.0",
            task.task_id,
        )
        return {
            "effective_tool_ratio": 1.0,
            "tool_call_total": 0,
            "tool_call_effective": 0,
            "tool_call_evaluations": [],
            "rule_invalid_count": 0,
            "llm_judged_count": 0,
            "orphan_tool_results_count": len(turn.orphan_tool_results()),
        }

    user_query = turn.user_normalized or ""
    slot: list[Optional[dict[str, Any]]] = [None] * T
    pending_events: list[ToolCallEvent] = []
    pending_refs: list[str] = []
    pending_indices: list[int] = []
    rule_invalid = 0
    prev_key: str | None = None

    for i, ev in enumerate(events):
        rule_tag, rule_reason = _rule_prefilter_event(ev, prev_call_key=prev_key)
        cur_key = f"{ev.name}\0{_args_key(ev.arguments)}"

        if rule_tag is not None:
            rule_invalid += 1
            slot[i] = {
                "annotator_ref": annotator_ref_for_event_index(i),
                "tool_call_id": ev.tool_call_id,
                "effective": False,
                **_rule_dim_fields_false(),
                "reason": rule_reason or rule_tag,
                "source": "rule",
                "rule": rule_tag,
            }
        else:
            ref = annotator_ref_for_event_index(i)
            pending_events.append(ev)
            pending_refs.append(ref)
            pending_indices.append(i)

        prev_key = cur_key

    llm_count = 0
    model = config.judge_model_names()[0]
    temp = config.tool_use_effectiveness_temperature()
    expected_refs = set(pending_refs)
    if pending_events:
        logger.debug(
            "Tool-effectiveness LLM annotation started task_id={} pending={} "
            "rule_invalid={} model={}",
            task.task_id,
            len(pending_events),
            rule_invalid,
            model,
        )
        parsed: dict[str, JudgmentTuple] = {}
        last_err: str | None = None
        try:
            missing_refs = set(expected_refs)
            for attempt in range(2):
                if attempt == 0:
                    req_refs = [r for r in pending_refs if r in missing_refs]
                    req_events = [ev for ev, r in zip(pending_events, pending_refs) if r in missing_refs]
                else:
                    if not missing_refs:
                        break
                    req_refs = [r for r in pending_refs if r in missing_refs]
                    req_events = [ev for ev, r in zip(pending_events, pending_refs) if r in missing_refs]
                req_expected = set(req_refs)
                if not req_refs:
                    break
                user_content = _build_annotator_user_payload(
                    task, user_query, turn, req_events, req_refs
                )
                if attempt > 0:
                    user_content = user_content + _annotate_user_repair_suffix(req_expected)
                raw = chat_completion_for_model(
                    [
                        {"role": "system", "content": TOOL_USE_ANNOTATOR_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    model=model,
                    temperature=temp,
                    response_format_json=True,
                )
                try:
                    req_id_fallback = _unique_tool_call_id_to_ref(req_events, req_refs)
                    parsed_cur = _parse_judgments(raw, req_expected, req_id_fallback)
                    parsed.update(parsed_cur)
                except Exception as e:
                    last_err = str(e)
                    logger.warning(
                        "Tool-effectiveness JSON parsing failed attempt={} err={}",
                        attempt + 1,
                        e,
                    )
                    if attempt == 0 and raw.strip():
                        try:
                            from eval.judging.client import _repair_json_text_loose

                            repaired = _repair_json_text_loose(raw.strip())
                            if repaired.startswith("```"):
                                repaired = repaired.split("\n", 1)[-1]
                            if repaired.rstrip().endswith("```"):
                                repaired = repaired.rsplit("```", 1)[0]
                            req_id_fallback = _unique_tool_call_id_to_ref(req_events, req_refs)
                            parsed_cur = _parse_judgments(repaired, req_expected, req_id_fallback)
                            parsed.update(parsed_cur)
                            last_err = None
                        except Exception:
                            pass
                missing_refs = expected_refs - set(parsed.keys())
                if not missing_refs:
                    break
                if attempt == 0:
                    last_err = f"缺少 judgment: {sorted(missing_refs)}"
                    logger.warning(
                        "Tool-effectiveness annotator_ref coverage incomplete: {}",
                        sorted(missing_refs),
                    )
        except Exception as e:
            logger.warning("Tool-effectiveness LLM annotation failed: {}", e)
            err_note = f"LLM 标注失败: {e}"
            for i, ev in enumerate(events):
                if slot[i] is not None:
                    continue
                slot[i] = {
                    "annotator_ref": annotator_ref_for_event_index(i),
                    "tool_call_id": ev.tool_call_id,
                    "effective": False,
                    **_rule_dim_fields_false(),
                    "reason": err_note,
                    "source": "llm_error",
                }
            ordered = [s for s in slot if s is not None]
            effective_n = sum(1 for x in ordered if x.get("effective"))
            ratio = effective_n / T if T else 1.0
            out = {
                "effective_tool_ratio": round(ratio, 6),
                "tool_call_total": T,
                "tool_call_effective": effective_n,
                "tool_call_evaluations": ordered,
                "rule_invalid_count": rule_invalid,
                "llm_judged_count": 0,
                "orphan_tool_results_count": len(turn.orphan_tool_results()),
                "annotator_model": model,
                "error": str(e),
            }
            logger.debug(
                "Tool-effectiveness LLM annotation ended with errors "
                "task_id={} ratio={} effective={}/{} err={}",
                task.task_id,
                out.get("effective_tool_ratio"),
                out.get("tool_call_effective"),
                out.get("tool_call_total"),
                out.get("error"),
            )
            return out

        llm_count = len(pending_events)
        for ref, ev, pi in zip(pending_refs, pending_events, pending_indices):
            pair = parsed.get(ref)
            if pair is None:
                tr, ac, ru, eff = False, False, False, False
                reason = last_err or "LLM 未返回该 annotator_ref 的 judgment"
            else:
                tr, ac, ru, eff, reason = pair
            slot[pi] = {
                "annotator_ref": ref,
                "tool_call_id": ev.tool_call_id,
                "task_relevance": tr,
                "action_coherence": ac,
                "result_usefulness": ru,
                "effective": eff,
                "reason": reason,
                "source": "llm",
            }

    ordered = [s for s in slot if s is not None]
    if len(ordered) != T:
        logger.error("Tool-effectiveness slots incomplete: T={} got={}", T, len(ordered))

    effective_n = sum(1 for x in ordered if x.get("effective"))
    ratio = effective_n / T if T else 1.0

    logger.debug(
        "Tool-effectiveness evaluation completed task_id={} "
        "effective_ratio={} effective={}/{} llm_judged={} rule_invalid={}",
        task.task_id,
        round(ratio, 6),
        effective_n,
        T,
        llm_count,
        rule_invalid,
    )
    return {
        "effective_tool_ratio": round(ratio, 6),
        "tool_call_total": T,
        "tool_call_effective": effective_n,
        "tool_call_evaluations": ordered,
        "rule_invalid_count": rule_invalid,
        "llm_judged_count": llm_count,
        "orphan_tool_results_count": len(turn.orphan_tool_results()),
        "annotator_model": model,
    }
