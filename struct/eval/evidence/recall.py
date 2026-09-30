"""Compute evidence recall from IDs returned in an agent's tool results."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from loguru import logger

from eval.core.tasks import TaskItem
from eval.evidence.catalog import (
    is_active_evidence_root,
    load_active_evidence_index,
)
from eval.traces.jsonl import ConversationTurn


_EVIDENCE_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])", re.I)
_ID_KEYS = frozenset({"id", "evidence_id", "evidenceId"})


def resolve_benchmark_data_root(
    explicit: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    value = explicit or os.environ.get("BENCHMARK_DATA_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else None


def resolve_index_user_id(tasks_path: Optional[Path]) -> str:
    configured = os.environ.get("BENCHMARK_INDEX_USER_ID", "").strip()
    if configured:
        return configured
    if tasks_path is not None:
        return tasks_path.resolve().parent.name
    return ""


def load_source_data_index(
    benchmark_data_root: Path,
    user_id: str = "",
    *,
    tasks_path: Optional[Path] = None,
    index_kind: str = "active",
) -> Optional[dict[str, Any]]:
    """Build the in-memory evidence index from active app ``batch.json`` files."""
    del tasks_path, index_kind
    root = Path(benchmark_data_root).expanduser().resolve()
    if any(part.lower() == "archive" for part in root.parts):
        return None
    if is_active_evidence_root(root):
        return load_active_evidence_index(root)
    nested = root / user_id if user_id else root
    if nested != root and is_active_evidence_root(nested):
        return load_active_evidence_index(nested)
    return None


def _walk_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _ID_KEYS and isinstance(child, (str, int)):
                candidate = str(child).strip().lower()
                if _EVIDENCE_ID_RE.fullmatch(candidate):
                    yield candidate
            elif isinstance(child, (dict, list)):
                yield from _walk_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_ids(child)


def _ids_from_tool_results(turn: ConversationTurn) -> set[str]:
    found: set[str] = set()
    for result in turn.tool_results:
        if isinstance(result.details, dict):
            found.update(_walk_ids(result.details))
        text = str(result.raw_content or "").strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        found.update(_walk_ids(parsed))
    return found


def compute_evidence_recall(
    turn: Optional[ConversationTurn],
    task: TaskItem,
    *,
    benchmark_data_root: Optional[Path] = None,
    index_user_id: Optional[str] = None,
    tasks_path: Optional[Path] = None,
    profile_override: Optional[str] = None,
    fuzzy: bool = False,
    fuzzy_threshold: float = 0.86,
    quiet: bool = False,
) -> dict[str, Any]:
    """Measure the fraction of gold evidence IDs present in tool outputs.

    Tool results must preserve ``id`` or ``evidence_id``. This explicit contract
    keeps the metric deterministic and independent of any agent implementation.
    """
    del profile_override, fuzzy, fuzzy_threshold
    gold = list(dict.fromkeys(str(item).lower() for item in task.evidence_ids or []))
    result: dict[str, Any] = {
        "skipped": False,
        "skip_reason": None,
        "gold_total": len(gold),
        "hit_count": 0,
        "recall": None,
        "recall_including_no_tool_call": None,
        "hits": [],
        "misses": [],
        "per_evidence": [],
        "index_user_id": index_user_id or resolve_index_user_id(tasks_path),
    }
    if not gold:
        result["skip_reason"] = "empty_gold"
        return result
    if turn is None:
        result.update(
            {
                "skip_reason": "no_matched_turn",
                "recall": 0.0,
                "recall_including_no_tool_call": 0.0,
                "misses": gold,
            }
        )
        result["per_evidence"] = [
            {"evidence_id": item, "matched": False, "reason": "no_matched_turn"}
            for item in gold
        ]
        return result
    if benchmark_data_root is None:
        result.update({"skipped": True, "skip_reason": "no_evidence_root"})
        return result

    index = load_source_data_index(
        benchmark_data_root,
        result["index_user_id"] or "",
        tasks_path=tasks_path,
    )
    if index is None:
        result.update({"skipped": True, "skip_reason": "invalid_evidence_root"})
        return result
    by_id = index.get("by_evidence_id")
    if not isinstance(by_id, dict):
        result.update({"skipped": True, "skip_reason": "invalid_evidence_index"})
        return result

    returned_ids = _ids_from_tool_results(turn)
    hits = [item for item in gold if item in returned_ids]
    misses = [item for item in gold if item not in returned_ids]
    has_tool_calls = bool(turn.iter_tool_call_events())
    recall = len(hits) / len(gold)
    result.update(
        {
            "hit_count": len(hits),
            "hits": hits,
            "misses": misses,
            "recall": recall if has_tool_calls else None,
            "recall_including_no_tool_call": recall,
            "per_evidence": [
                {
                    "evidence_id": item,
                    "matched": item in returned_ids,
                    "data_type": (
                        by_id.get(item, {}).get("data_type")
                        if isinstance(by_id.get(item), dict)
                        else None
                    ),
                    "reason": "evidence_id_in_tool_result"
                    if item in returned_ids
                    else "evidence_id_not_in_tool_result",
                }
                for item in gold
            ],
            "debug": {
                "has_tool_calls": has_tool_calls,
                "tool_call_events_count": len(turn.iter_tool_call_events()),
                "returned_evidence_id_count": len(returned_ids),
            },
        }
    )
    if not quiet:
        logger.debug(
            "Evidence recall task_id={} hit={}/{}", task.task_id, len(hits), len(gold)
        )
    return result
