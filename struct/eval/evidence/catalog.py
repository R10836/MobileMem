"""Build and validate an evidence catalog from active structured app data.

Only explicitly supported live paths are read.  ``archive`` directories,
review documents, and ``*_rad``/``*_con`` variants are deliberately outside
this module's discovery rules.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from eval.core.tasks import TaskItem


EVIDENCE_ID_RE = re.compile(r"^[0-9a-f]{12}$", re.IGNORECASE)
_GT_ID_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵]")
_QUERY_DATE_RE = re.compile(
    r"当前时间(?:为|是)?\s*[:：]?\s*(?P<year>\d{4})\s*(?:年|[-/.])\s*"
    r"(?P<month>\d{1,2})\s*(?:月|[-/.])\s*(?P<day>\d{1,2})\s*日?"
)

CORE_SOURCES: tuple[tuple[str, str], ...] = (
    ("bill/batch.json", "bill"),
    ("voice/batch.json", "voice_memo"),
    ("note/batch.json", "note"),
    ("document/batch.json", "document"),
    ("calendar/batch.json", "calendar"),
    ("todo/batch.json", "todo"),
)

# Optional sources are intentionally explicit.  Adding a new source requires
# a reviewed adapter here; recursive JSON discovery would accidentally ingest
# generation/review artifacts.
OPTIONAL_JSON_SOURCES: tuple[tuple[str, str], ...] = (
    ("video/description.json", "video"),
    ("image/description.json", "image"),
    ("event/user.json", "persona_event"),
)

_ACTIVE_INDEX_CACHE: dict[
    str, tuple[tuple[tuple[str, int, int], ...], dict[str, Any]]
] = {}
_ACTIVE_INDEX_CACHE_LOCK = threading.RLock()


class EvidenceCatalogError(ValueError):
    """Raised for malformed or conflicting active evidence."""


class EvidencePreflightError(ValueError):
    """Raised when tasks cannot be safely evaluated against active evidence."""

    def __init__(self, report: "EvidencePreflightReport") -> None:
        self.report = report
        preview = "; ".join(
            f"{issue.task_id}:{issue.code}:{issue.message}" for issue in report.errors[:8]
        )
        if len(report.errors) > 8:
            preview += f"; plus {len(report.errors) - 8} more"
        super().__init__(f"Evidence preflight failed ({len(report.errors)} issues): {preview}")


@dataclass(frozen=True)
class EvidenceIssue:
    task_id: str
    code: str
    message: str
    evidence_id: str | None = None


@dataclass
class EvidencePreflightReport:
    evidence_root: str
    task_count: int
    evidence_count: int
    errors: list[EvidenceIssue] = field(default_factory=list)
    warnings: list[EvidenceIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        def item(x: EvidenceIssue) -> dict[str, Any]:
            return {
                "task_id": x.task_id,
                "code": x.code,
                "message": x.message,
                "evidence_id": x.evidence_id,
            }

        return {
            "ok": self.ok,
            "evidence_root": self.evidence_root,
            "task_count": self.task_count,
            "evidence_count": self.evidence_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [item(x) for x in self.errors],
            "warnings": [item(x) for x in self.warnings],
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value:
            return
        self.parts.append(value)
        if self._in_title:
            self.title_parts.append(value)


def is_active_evidence_root(path: Path) -> bool:
    root = Path(path)
    if any(part.lower() == "archive" for part in root.parts):
        return False
    return all((root / relative).is_file() for relative, _ in CORE_SOURCES)


def _record_id(record: dict[str, Any]) -> str | None:
    for key in ("evidence_id", "id"):
        value = record.get(key)
        if value is not None:
            candidate = str(value).strip()
            if EVIDENCE_ID_RE.fullmatch(candidate):
                return candidate.lower()
    return None


def _iter_explicit_id_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if _record_id(value):
            yield value
            return
        for child in value.values():
            yield from _iter_explicit_id_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_explicit_id_records(child)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceCatalogError(f"Cannot read evidence file {path}: {exc}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _add_entry(
    by_id: dict[str, dict[str, Any]],
    sources: dict[str, str],
    *,
    evidence_id: str,
    data_type: str,
    record: dict[str, Any],
    source: Path,
) -> None:
    entry = {"data_type": data_type, "record": record}
    previous = by_id.get(evidence_id)
    if previous is not None:
        if _canonical_json(previous) == _canonical_json(entry):
            return
        raise EvidenceCatalogError(
            f"Conflicting content for evidence_id {evidence_id}: "
            f"{sources[evidence_id]} versus {source}"
        )
    by_id[evidence_id] = entry
    sources[evidence_id] = str(source.resolve())


def _load_array_source(
    path: Path,
    fallback_type: str,
    by_id: dict[str, dict[str, Any]],
    sources: dict[str, str],
    *,
    recursive: bool = False,
) -> None:
    value = _read_json(path)
    records: Iterable[dict[str, Any]]
    if recursive:
        records = _iter_explicit_id_records(value)
    elif isinstance(value, list):
        records = (item for item in value if isinstance(item, dict))
    else:
        raise EvidenceCatalogError(f"Evidence file must contain a top-level array: {path}")
    for record in records:
        evidence_id = _record_id(record)
        if evidence_id is None:
            if recursive:
                continue
            raise EvidenceCatalogError(
                f"Evidence record lacks a 12-character id/evidence_id: {path}"
            )
        data_type = str(record.get("data_type") or fallback_type).strip() or fallback_type
        _add_entry(
            by_id,
            sources,
            evidence_id=evidence_id,
            data_type=data_type,
            record=record,
            source=path,
        )


def _load_screen_sources(
    root: Path,
    by_id: dict[str, dict[str, Any]],
    sources: dict[str, str],
) -> list[Path]:
    used: list[Path] = []
    screen_dir = root / "screen"
    if not screen_dir.is_dir():
        return used
    for path in sorted(screen_dir.glob("*.html")):
        evidence_id = path.stem.lower()
        if not EVIDENCE_ID_RE.fullmatch(evidence_id):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(raw)
        record = {
            "id": evidence_id,
            "data_type": "screen",
            "title": " ".join(parser.title_parts),
            "text": "\n".join(parser.parts),
            "source_path": str(path.resolve()),
        }
        _add_entry(
            by_id,
            sources,
            evidence_id=evidence_id,
            data_type="screen",
            record=record,
            source=path,
        )
        used.append(path)
    return used


def load_active_evidence_index(root: Path) -> dict[str, Any]:
    """Build an in-memory index for one user's active evidence directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    if not is_active_evidence_root(root):
        raise EvidenceCatalogError(
            "Not an active evidence directory; six required batch.json files "
            f"were not found: {root}"
        )

    candidate_paths = [
        root / relative
        for relative, _ in (*CORE_SOURCES, *OPTIONAL_JSON_SOURCES)
        if (root / relative).is_file()
    ]
    screen_dir = root / "screen"
    if screen_dir.is_dir():
        candidate_paths.extend(sorted(screen_dir.glob("*.html")))
    signature = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in candidate_paths
    )
    cache_key = str(root)
    with _ACTIVE_INDEX_CACHE_LOCK:
        cached = _ACTIVE_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    by_id: dict[str, dict[str, Any]] = {}
    source_for_id: dict[str, str] = {}
    loaded_paths: list[Path] = []
    for relative, data_type in CORE_SOURCES:
        path = root / relative
        if not path.is_file():
            continue
        _load_array_source(path, data_type, by_id, source_for_id)
        loaded_paths.append(path)

    for relative, data_type in OPTIONAL_JSON_SOURCES:
        path = root / relative
        if not path.is_file():
            continue
        _load_array_source(
            path,
            data_type,
            by_id,
            source_for_id,
            recursive=True,
        )
        loaded_paths.append(path)
    loaded_paths.extend(_load_screen_sources(root, by_id, source_for_id))

    result = {
        "schema_version": 3,
        "user_id": _infer_user_id(by_id),
        "by_evidence_id": by_id,
        "meta": {
            "source": "active_app_files",
            "evidence_root": str(root),
            "loaded_paths": [str(p.resolve()) for p in loaded_paths],
            "archive_ignored": True,
        },
    }
    with _ACTIVE_INDEX_CACHE_LOCK:
        _ACTIVE_INDEX_CACHE[cache_key] = (signature, result)
    return result


def _infer_user_id(by_id: dict[str, dict[str, Any]]) -> str | None:
    for entry in by_id.values():
        record = entry.get("record")
        if isinstance(record, dict) and record.get("user_id") is not None:
            return str(record["user_id"])
    return None


def _gt_count(text: str) -> int:
    return len(_GT_ID_RE.findall(str(text or "")))


def _query_date(text: str) -> date | None:
    match = _QUERY_DATE_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _created_date(record: dict[str, Any]) -> date | None:
    raw = next(
        (
            record.get(key)
            for key in ("created_at", "创建时间", "create_time")
            if record.get(key) is not None
        ),
        None,
    )
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        match = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        if not match:
            return None
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None


def validate_tasks_against_index(
    tasks: Sequence[TaskItem],
    index: dict[str, Any],
    *,
    evidence_root: Path,
) -> EvidencePreflightReport:
    by_id = index.get("by_evidence_id")
    if not isinstance(by_id, dict):
        raise EvidenceCatalogError("Active evidence index lacks by_evidence_id")
    report = EvidencePreflightReport(
        evidence_root=str(Path(evidence_root).resolve()),
        task_count=len(tasks),
        evidence_count=len(by_id),
    )
    for task in tasks:
        evidence_ids = [str(x).strip().lower() for x in (task.evidence_ids or []) if str(x).strip()]
        if len(evidence_ids) != len(set(evidence_ids)):
            report.errors.append(
                EvidenceIssue(
                    task.task_id,
                    "duplicate_task_evidence_id",
                    "Duplicate evidence_id within task",
                )
            )
        gt_count = _gt_count(task.metadata_gt or "")
        if gt_count != len(evidence_ids):
            report.errors.append(
                EvidenceIssue(
                    task.task_id,
                    "gt_evidence_count_mismatch",
                    f"GT contains {gt_count} items but evidence_ids contains {len(evidence_ids)}",
                )
            )

        query_date = _query_date(task.prompt)
        for evidence_id in dict.fromkeys(evidence_ids):
            entry = by_id.get(evidence_id)
            if not isinstance(entry, dict):
                report.errors.append(
                    EvidenceIssue(
                        task.task_id,
                        "missing_evidence_id",
                        "ID does not exist in the active evidence directory",
                        evidence_id,
                    )
                )
                continue
            record = entry.get("record")
            if not isinstance(record, dict):
                report.errors.append(
                    EvidenceIssue(
                        task.task_id,
                        "invalid_evidence_record",
                        "Evidence record is not an object",
                        evidence_id,
                    )
                )
                continue
            if query_date is not None:
                created = _created_date(record)
                if created is None:
                    report.warnings.append(
                        EvidenceIssue(
                            task.task_id,
                            "created_at_unavailable",
                            "Query has a current time but evidence has no parseable created_at",
                            evidence_id,
                        )
                    )
                elif created > query_date:
                    report.errors.append(
                        EvidenceIssue(
                            task.task_id,
                            "evidence_not_visible_at_query_time",
                            f"created_at={created.isoformat()} is later than "
                            f"Query date {query_date.isoformat()}",
                            evidence_id,
                        )
                    )
    return report


def preflight_active_evidence(
    tasks: Sequence[TaskItem],
    evidence_root: Path,
) -> tuple[dict[str, Any], EvidencePreflightReport]:
    index = load_active_evidence_index(evidence_root)
    report = validate_tasks_against_index(tasks, index, evidence_root=evidence_root)
    if not report.ok:
        raise EvidencePreflightError(report)
    return index, report
