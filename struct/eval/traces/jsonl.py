"""Parse generic OpenAI-style JSONL agent traces into evaluation turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_DATE_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_TIME_ANCHOR_RE = re.compile(
    r"^(?:（当前时间(?:为)?[^）]+）|当前时间(?:为)?[^，,。\s]+[，,]?\s*)"
)
_RUNTIME_CONTEXT_RE = re.compile(r"<runtime_context>.*?</runtime_context>\s*", re.S | re.I)
_USER_INSTRUCTION_RE = re.compile(
    r"<user_instruction>\s*(.*?)\s*</user_instruction>", re.S | re.I
)
_MATCH_IGNORE_PUNCT_RE = re.compile(
    r"[\u3000-\u303f\uff00-\uff65、，,。！？；：\"\"''（）()\[\]{}]"
)


def _strip_sender_metadata(raw: str) -> str:
    if "Sender (untrusted metadata)" not in raw:
        return raw
    index = raw.find("Sender (untrusted metadata)")
    tail = raw[index:]
    first_fence = tail.find("```")
    fence_end = tail.find("```", first_fence + 3) if first_fence >= 0 else -1
    return tail[fence_end + 3 :].lstrip() if fence_end >= 0 else raw


def _extract_instruction_text(raw: str) -> str:
    match = _USER_INSTRUCTION_RE.search(raw)
    if match:
        return match.group(1).strip()
    return _RUNTIME_CONTEXT_RE.sub("", raw).strip()


def normalize_user_text(text: str) -> str:
    text = _strip_sender_metadata(text)
    if "<user_instruction>" in text.lower() or "<runtime_context>" in text.lower():
        text = _extract_instruction_text(text)
    text = _DATE_PREFIX_RE.sub("", text.strip())
    text = _TIME_ANCHOR_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_query_match_text(text: str) -> str:
    """Normalize harmless formatting differences for Query-to-turn matching."""
    return _MATCH_IGNORE_PUNCT_RE.sub("", normalize_user_text(text)).strip()


def is_business_user_message(normalized_text: str) -> bool:
    return bool(normalized_text)


def _parse_timestamp(
    obj: dict[str, Any], inner: Optional[dict[str, Any]] = None
) -> Optional[datetime]:
    candidates = [obj.get("timestamp")]
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("timestamp"))
    if inner:
        candidates.append(inner.get("timestamp"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                continue
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000.0 if value > 1e12 else float(value)
            try:
                return datetime.fromtimestamp(seconds, tz=timezone.utc)
            except (OSError, ValueError, OverflowError):
                continue
    return None


def _content_part_kind(part: dict[str, Any]) -> Optional[str]:
    kind = part.get("type") or part.get("role")
    return str(kind) if kind in ("text", "toolCall", "thinking") else None


def _text_from_content_part(part: dict[str, Any]) -> str:
    if _content_part_kind(part) != "text":
        return ""
    return str(part.get("text") or "").strip()


def extract_user_text_from_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return _extract_instruction_text(content)
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                value = _text_from_content_part(block)
                if value:
                    parts.append(value)
    return "\n".join(parts).strip()


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_assistant_message(obj: dict[str, Any]) -> dict[str, Any]:
    content = obj.get("content")
    parts: list[dict[str, Any]] = (
        [dict(part) for part in content if isinstance(part, dict)]
        if isinstance(content, list)
        else []
    )
    if not isinstance(content, list):
        reasoning = obj.get("reasoning_content")
        if reasoning and str(reasoning).strip():
            parts.append({"type": "thinking", "thinking": str(reasoning).strip()})
        if content and str(content).strip():
            parts.append({"type": "text", "text": str(content).strip()})
    existing_call_ids = {
        str(part.get("id"))
        for part in parts
        if _content_part_kind(part) == "toolCall" and part.get("id")
    }
    for tool_call in obj.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = str(tool_call.get("id") or "")
        if tool_call_id and tool_call_id in existing_call_ids:
            continue
        function = tool_call.get("function")
        function = function if isinstance(function, dict) else {}
        parts.append(
            {
                "type": "toolCall",
                "id": tool_call_id or None,
                "name": str(function.get("name") or tool_call.get("name") or ""),
                "arguments": _parse_tool_arguments(
                    function.get("arguments", tool_call.get("arguments"))
                ),
            }
        )
        if tool_call_id:
            existing_call_ids.add(tool_call_id)
    message = {"role": "assistant", "content": parts}
    if isinstance(obj.get("usage"), dict):
        message["usage"] = obj["usage"]
    return message


@dataclass
class ToolCallRecord:
    tool_call_id: Optional[str]
    name: str
    arguments: dict


@dataclass
class ToolResultRecord:
    tool_call_id: Optional[str]
    tool_name: Optional[str]
    raw_content: str
    is_error: bool
    details: Optional[dict] = None


def _tool_result_from_message(message: dict[str, Any]) -> ToolResultRecord:
    content = message.get("content")
    if isinstance(content, str):
        raw_content = content
    elif content is None:
        raw_content = ""
    else:
        raw_content = json.dumps(content, ensure_ascii=False)
    details = message.get("details")
    return ToolResultRecord(
        tool_call_id=str(
            message.get("tool_call_id") or message.get("toolCallId") or ""
        )
        or None,
        tool_name=str(message.get("name") or message.get("toolName") or "") or None,
        raw_content=raw_content,
        is_error=bool(message.get("is_error") or message.get("isError")),
        details=details if isinstance(details, dict) else None,
    )


@dataclass
class ToolCallEvent:
    tool_call_id: Optional[str]
    name: str
    arguments: dict[str, Any]
    result: Optional[ToolResultRecord]
    thinking_before_call: str
    thinking_message_full: str


def _tool_results_by_call_id(
    tool_results: List[ToolResultRecord],
) -> Dict[str, ToolResultRecord]:
    result: Dict[str, ToolResultRecord] = {}
    for item in tool_results:
        if item.tool_call_id:
            result[str(item.tool_call_id)] = item
    return result


def _tool_call_events_from_assistant_message(
    message: dict[str, Any], by_id: Dict[str, ToolResultRecord]
) -> List[ToolCallEvent]:
    parts = message.get("content") or []
    if not isinstance(parts, list):
        return []
    thinking_parts = [
        str(part.get("thinking") or part.get("text") or "").strip()
        for part in parts
        if isinstance(part, dict) and _content_part_kind(part) == "thinking"
    ]
    thinking_full = "\n\n".join(value for value in thinking_parts if value)
    events: list[ToolCallEvent] = []
    before: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = _content_part_kind(part)
        if kind == "thinking":
            value = str(part.get("thinking") or part.get("text") or "").strip()
            if value:
                before.append(value)
            continue
        if kind != "toolCall":
            continue
        tool_call_id = str(part.get("id") or "") or None
        arguments = part.get("arguments")
        events.append(
            ToolCallEvent(
                tool_call_id=tool_call_id,
                name=str(part.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                result=by_id.get(tool_call_id) if tool_call_id else None,
                thinking_before_call="\n\n".join(before).strip(),
                thinking_message_full=thinking_full,
            )
        )
        before = []
    return events


@dataclass
class ConversationTurn:
    user_raw: str
    user_normalized: str
    is_business: bool
    assistant_blocks: List[dict] = field(default_factory=list)
    tool_results: List[ToolResultRecord] = field(default_factory=list)
    source_log_path: Optional[str] = None
    turn_start_time: Optional[datetime] = None
    turn_latest_time: Optional[datetime] = None

    def final_assistant_text(self) -> str:
        for block in reversed(self.assistant_blocks):
            message = block.get("message", block)
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            texts = (
                [
                    _text_from_content_part(part)
                    for part in content
                    if isinstance(part, dict)
                ]
                if isinstance(content, list)
                else []
            )
            answer = "\n".join(value for value in texts if value).strip()
            if answer:
                return answer
        return ""

    def tool_results_by_call_id(self) -> Dict[str, ToolResultRecord]:
        return _tool_results_by_call_id(self.tool_results)

    def iter_tool_call_events(self) -> list[ToolCallEvent]:
        by_id = self.tool_results_by_call_id()
        events: list[ToolCallEvent] = []
        for block in self.assistant_blocks:
            message = block.get("message", block)
            if isinstance(message, dict):
                events.extend(_tool_call_events_from_assistant_message(message, by_id))
        if len(events) == 1 and events[0].result is None and not events[0].tool_call_id:
            unmatched = [item for item in self.tool_results if not item.tool_call_id]
            if len(unmatched) == 1:
                events[0].result = unmatched[0]
        return events

    def iter_tool_calls_in_order(self) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(event.tool_call_id, event.name, event.arguments)
            for event in self.iter_tool_call_events()
        ]

    def orphan_tool_results(self) -> list[ToolResultRecord]:
        paired = {id(event.result) for event in self.iter_tool_call_events() if event.result}
        return [item for item in self.tool_results if id(item) not in paired]

    def agent_total_tokens(self) -> int:
        total = 0
        for block in self.assistant_blocks:
            message = block.get("message", block)
            usage = message.get("usage") if isinstance(message, dict) else None
            if not isinstance(usage, dict):
                continue
            value = usage.get("total_tokens", usage.get("totalTokens"))
            if value is not None:
                try:
                    total += max(0, int(value))
                    continue
                except (TypeError, ValueError):
                    pass
            for key in ("prompt_tokens", "completion_tokens", "input", "output"):
                try:
                    total += max(0, int(usage.get(key, 0) or 0))
                except (TypeError, ValueError):
                    continue
        return total

    def agent_processing_duration_seconds(self) -> Optional[float]:
        if self.turn_start_time is None or self.turn_latest_time is None:
            return None
        return max(0.0, (self.turn_latest_time - self.turn_start_time).total_seconds())


@dataclass
class ParsedSession:
    session_id: Optional[str]
    cwd: Optional[str]
    turns: List[ConversationTurn]
    source_paths: Optional[List[str]] = None

    def business_turns(self) -> List[tuple[int, ConversationTurn]]:
        return [(index, turn) for index, turn in enumerate(self.turns) if turn.is_business]


def _bump_turn_latest_time(turn: ConversationTurn, value: Optional[datetime]) -> None:
    if value is not None and (turn.turn_latest_time is None or value > turn.turn_latest_time):
        turn.turn_latest_time = value


def _start_user_turn(
    turns: List[ConversationTurn],
    current: Optional[ConversationTurn],
    raw: str,
    timestamp: Optional[datetime],
) -> ConversationTurn:
    if current is not None:
        turns.append(current)
    normalized = normalize_user_text(raw)
    return ConversationTurn(
        user_raw=raw,
        user_normalized=normalized,
        is_business=is_business_user_message(normalized),
        turn_start_time=timestamp,
        turn_latest_time=timestamp,
    )


def _message_from_record(obj: dict[str, Any]) -> Optional[dict[str, Any]]:
    if obj.get("type") == "message" and isinstance(obj.get("message"), dict):
        return obj["message"]
    if obj.get("role") in {"user", "assistant", "tool", "toolResult"}:
        return obj
    return None


def parse_jsonl_session(path: Union[str, Path]) -> ParsedSession:
    path = Path(path)
    session_id: Optional[str] = None
    cwd: Optional[str] = None
    turns: List[ConversationTurn] = []
    current: Optional[ConversationTurn] = None

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")

            if obj.get("_type") == "metadata":
                session_id = session_id or (
                    str(obj.get("key")) if obj.get("key") else None
                )
                continue
            if obj.get("type") == "session":
                session_id = str(obj.get("id") or "") or session_id
                cwd = str(obj.get("cwd") or "") or cwd
                continue

            message = _message_from_record(obj)
            if message is None:
                continue
            role = message.get("role")
            timestamp = _parse_timestamp(obj, message)
            if role == "user":
                current = _start_user_turn(
                    turns, current, extract_user_text_from_message(message), timestamp
                )
            elif current is None:
                continue
            elif role == "assistant":
                _bump_turn_latest_time(current, timestamp)
                current.assistant_blocks.append(
                    {"message": _normalize_assistant_message(message)}
                )
            elif role in {"tool", "toolResult"}:
                _bump_turn_latest_time(current, timestamp)
                current.tool_results.append(_tool_result_from_message(message))

    if current is not None:
        turns.append(current)
    absolute_path = str(path.resolve())
    for turn in turns:
        turn.source_log_path = absolute_path
    return ParsedSession(
        session_id=session_id,
        cwd=cwd,
        turns=turns,
        source_paths=[absolute_path],
    )


def parse_directory_jsonl_sessions(dir_path: Union[str, Path]) -> ParsedSession:
    directory = Path(dir_path)
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {directory.resolve()}")

    turns: List[ConversationTurn] = []
    source_paths: List[str] = []
    session_ids: List[str] = []
    first_cwd: Optional[str] = None
    for file_path in files:
        session = parse_jsonl_session(file_path)
        turns.extend(session.turns)
        source_paths.extend(session.source_paths or [])
        if session.session_id:
            session_ids.append(session.session_id)
        first_cwd = first_cwd or session.cwd
    return ParsedSession(
        session_id=",".join(session_ids) or None,
        cwd=first_cwd,
        turns=turns,
        source_paths=source_paths,
    )
