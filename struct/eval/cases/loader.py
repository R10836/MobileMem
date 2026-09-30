"""Load benchmark cases from UTF-8 CSV files."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from eval.core.tasks import LoadedTasksDocument, TaskItem


CASE_COLUMNS = (
    "场景维度",
    "Query编号",
    "用户场景",
    "Query",
    "evidence_ids",
    "GT",
    "gold answer",
    "得分点",
    "扣分点",
    "通过阈值",
    "对应理论维度",
    "适用能力",
)


def parse_pass_threshold(value: Any, *, default: float = 0.8) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", text)
    if not match:
        raise ValueError(f"Invalid pass threshold: {value!r}")
    threshold = float(match.group(1))
    if threshold > 1:
        threshold /= 100
    if not 0 <= threshold <= 1:
        raise ValueError(f"Pass threshold must be between 0 and 1: {value!r}")
    return threshold


def parse_evidence_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"无", "-", "n/a", "na", "none"}:
        return []
    return [part for part in re.split(r"[,，;；\s]+", text) if part]


def _require_columns(fieldnames: Iterable[str] | None, source: Path) -> None:
    found = {str(name or "").strip() for name in fieldnames or []}
    missing = [name for name in CASE_COLUMNS if name not in found]
    if missing:
        raise ValueError(f"Missing CSV columns in {source}: {', '.join(missing)}")


def _row_to_task(row: dict[str, str], *, source: Path, row_number: int) -> TaskItem:
    task_id = str(row.get("Query编号") or "").strip()
    prompt = str(row.get("Query") or "").strip()
    if not task_id:
        raise ValueError(f"Missing Query编号 at {source}:{row_number}")
    if not prompt:
        raise ValueError(f"Missing Query at {source}:{row_number}")

    evidence_ids = parse_evidence_ids(row.get("evidence_ids"))
    positive = str(row.get("得分点") or "").strip()
    deductions = str(row.get("扣分点") or "").strip()
    criteria = positive
    if deductions:
        criteria = f"{criteria}\n\n【扣分标准】\n{deductions}" if criteria else deductions

    scene = str(row.get("用户场景") or "").strip()
    capability = str(row.get("场景维度") or "").strip()
    return TaskItem(
        task_id=task_id,
        name=scene or task_id,
        capability=capability,
        sub_capability=str(row.get("适用能力") or "").strip(),
        dimension=str(row.get("对应理论维度") or "").strip(),
        query_type="",
        data_sources=",".join(evidence_ids),
        prompt=prompt,
        expected_behavior=str(row.get("gold answer") or "").strip(),
        grading_criteria=criteria,
        pass_threshold=parse_pass_threshold(row.get("通过阈值")),
        query_match="normalized",
        evidence_ids=evidence_ids or None,
        metadata_gt=str(row.get("GT") or "").strip() or None,
    )


def load_csv_tasks_document(path: Path) -> LoadedTasksDocument:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))

    tasks: list[TaskItem] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, source)
        for row_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            task = _row_to_task(row, source=source, row_number=row_number)
            if task.task_id in seen:
                raise ValueError(
                    f"Duplicate Query编号 {task.task_id!r} at {source}:{row_number}"
                )
            seen.add(task.task_id)
            tasks.append(task)

    if not tasks:
        raise ValueError(f"No benchmark cases found in {source}")
    logger.info("Loaded {} cases from {}", len(tasks), source)
    return LoadedTasksDocument(
        log_path=None,
        schema_version=1,
        tasks=tasks,
        format="csv_cases",
        load_stats={"rows": len(tasks)},
    )
