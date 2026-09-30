"""Resolve and load JSONL agent traces."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

from loguru import logger

from eval.traces.jsonl import (
    ParsedSession,
    parse_directory_jsonl_sessions,
    parse_jsonl_session,
)


def resolve_log_target(base_dir: Path, spec: Union[str, Path]) -> Path:
    path = Path(spec)
    path = path.resolve() if path.is_absolute() else (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent trace does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"Agent trace is neither a file nor a directory: {path}")
    return path


def resolve_log_target_prefer_bases(
    bases: Sequence[Path], spec: Union[str, Path]
) -> Tuple[Path, Path]:
    candidates: List[Path] = [base.resolve() for base in bases]
    if not candidates:
        raise ValueError("At least one trace base directory is required")
    raw = Path(spec)
    if raw.is_absolute():
        return resolve_log_target(candidates[0], raw), candidates[0]
    last_error: FileNotFoundError | None = None
    for base in candidates:
        try:
            return resolve_log_target(base, raw), base
        except FileNotFoundError as exc:
            last_error = exc
    raise FileNotFoundError(
        f"Agent trace {str(spec)!r} was not found under: "
        f"{[str(base) for base in candidates]}"
    ) from last_error


def load_parsed_session_from_spec(
    base_dir: Path, spec: Union[str, Path]
) -> ParsedSession:
    return load_parsed_session_resolved(resolve_log_target(base_dir, spec))


def load_parsed_session_resolved(path: Path) -> ParsedSession:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent trace does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Agent trace must be JSONL: {path}")
        logger.debug("Loading JSONL trace: {}", path)
        return parse_jsonl_session(path)
    if path.is_dir():
        logger.debug("Loading JSONL trace directory: {}", path)
        return parse_directory_jsonl_sessions(path)
    raise ValueError(f"Agent trace is neither a file nor a directory: {path}")
