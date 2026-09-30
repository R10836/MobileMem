"""Benchmark task models, loading, and trace-turn matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from eval.traces.jsonl import (
    ConversationTurn,
    ParsedSession,
    normalize_query_match_text,
)


QueryMatchMode = Literal["exact", "contains", "normalized"]


def _coerce_data_sources(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return "" if value is None else str(value).strip()


class TaskItem(BaseModel):
    """One evaluation case after normalization from CSV."""

    task_id: str
    name: str
    capability: str
    sub_capability: str
    dimension: str
    query_type: str
    data_sources: str
    prompt: str
    expected_behavior: str
    grading_criteria: str
    pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    query_match: QueryMatchMode = "normalized"
    turn_index: Optional[int] = None
    session_log: Optional[str] = None
    evidence_ids: Optional[List[str]] = None
    supporting_artifact_ids: Optional[List[str]] = None
    benchmark_user_id: Optional[str] = None
    metadata_gt: Optional[str] = None

    @field_validator("data_sources", mode="before")
    @classmethod
    def _normalize_data_sources(cls, value: Any) -> str:
        return _coerce_data_sources(value)

    def prompt_normalized(self) -> str:
        return normalize_query_match_text(self.prompt)


@dataclass
class LoadedTasksDocument:
    log_path: Optional[str]
    schema_version: int
    tasks: List[TaskItem]
    format: Literal["csv_cases"]
    load_stats: Optional[dict[str, Any]] = None
    skipped_empty_prompt: int = 0


def load_tasks_document(path: Union[str, Path]) -> LoadedTasksDocument:
    """Load the public ``case.csv`` format."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported case format; expected .csv: {source}")
    from eval.cases.loader import load_csv_tasks_document

    return load_csv_tasks_document(source)


def _turn_has_tool_result_for_disambiguation(turn: ConversationTurn) -> bool:
    return bool(turn.tool_results) or any(
        event.result is not None for event in turn.iter_tool_call_events()
    )


def _approx_tool_result_text_chars(turn: ConversationTurn) -> int:
    total = 0
    for result in turn.tool_results:
        total += len(result.raw_content or "")
        if result.details:
            total += len(json.dumps(result.details, ensure_ascii=False, default=str))
    return total


def _prompt_matches_turn(mode: QueryMatchMode, prompt: str, user: str) -> bool:
    if mode == "exact":
        return prompt == user
    prompt = normalize_query_match_text(prompt)
    user = normalize_query_match_text(user)
    if not prompt or not user:
        return False
    if mode == "normalized":
        return prompt == user
    return prompt in user or user in prompt


def query_texts_match(
    mode: QueryMatchMode, prompt_a: str, prompt_b: str
) -> bool:
    return _prompt_matches_turn(mode, prompt_a, prompt_b)


@dataclass
class MatchResult:
    task: TaskItem
    turn: Optional[ConversationTurn]
    global_turn_index: Optional[int]
    business_turn_index: Optional[int]
    status: Literal["ok", "skipped", "error"]
    warnings: List[str]


def match_task_to_session(task: TaskItem, session: ParsedSession) -> MatchResult:
    business_turns = session.business_turns()
    if task.turn_index is not None:
        if task.turn_index < 0 or task.turn_index >= len(business_turns):
            return MatchResult(
                task=task,
                turn=None,
                global_turn_index=None,
                business_turn_index=None,
                status="error",
                warnings=[f"turn_index {task.turn_index} is out of range"],
            )
        global_index, turn = business_turns[task.turn_index]
        return MatchResult(
            task=task,
            turn=turn,
            global_turn_index=global_index,
            business_turn_index=task.turn_index,
            status="ok",
            warnings=[],
        )

    matches = [
        (index, turn)
        for index, turn in enumerate(session.turns)
        if turn.is_business
        and _prompt_matches_turn(task.query_match, task.prompt, turn.user_raw)
    ]
    if not matches:
        return MatchResult(
            task=task,
            turn=None,
            global_turn_index=None,
            business_turn_index=None,
            status="skipped",
            warnings=["No matching user turn was found"],
        )

    warnings: List[str] = []
    if len(matches) > 1:
        ranked = [
            (index, turn, _approx_tool_result_text_chars(turn))
            for index, turn in matches
        ]
        with_tools = [
            item for item in ranked if _turn_has_tool_result_for_disambiguation(item[1])
        ]
        selected = min(with_tools or ranked, key=lambda item: item[0])
        global_index, turn, result_chars = selected
        warnings.append(
            f"Query matched {len(matches)} turns; selected the earliest suitable turn "
            f"(global_turn_index={global_index}, tool_text_chars≈{result_chars})"
        )
    else:
        global_index, turn = matches[0]
    business_index = next(
        index for index, (candidate, _) in enumerate(business_turns)
        if candidate == global_index
    )
    return MatchResult(
        task=task,
        turn=turn,
        global_turn_index=global_index,
        business_turn_index=business_index,
        status="ok",
        warnings=warnings,
    )


def match_all_tasks(
    tasks: List[TaskItem], session: ParsedSession
) -> List[MatchResult]:
    return [match_task_to_session(task, session) for task in tasks]


def match_tasks_to_turns(
    tasks: List[TaskItem], session: ParsedSession
) -> List[MatchResult]:
    return match_all_tasks(tasks, session)
