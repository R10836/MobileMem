"""Schemas for Judge output and evaluation reports."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CheckpointEval(BaseModel):
    id: str = ""
    reason: str = ""
    max_points: float = 0.0
    passed: bool = False
    score: float = 0.0
    # Required for memory-retrieval checkpoints.  The evaluator uses these GT
    # labels to recompute the equal-weight score deterministically.
    matched_gt_ids: List[str] = Field(default_factory=list)


class DeductionEval(BaseModel):
    reason: str = ""
    points: float = 0.0
    max_points: Optional[float] = None
    unit_points: Optional[float] = None
    occurrences: int = Field(default=0, ge=0)


class JudgeOutput(BaseModel):
    checkpoints: List[CheckpointEval] = Field(default_factory=list)
    deductions: List[DeductionEval] = Field(default_factory=list)
    total_score_0_100: float = 0.0
    passed: bool = False
    rationale: str = ""


def recompute_score_from_checkpoints(out: JudgeOutput) -> float:
    """Return checkpoint points minus deductions, without score clamping."""
    earned = sum(c.score for c in out.checkpoints)
    deduct = sum(d.points for d in out.deductions)
    return earned - deduct


def normalize_judge_output(out: JudgeOutput, *, pass_threshold: float) -> JudgeOutput:
    """Recompute the total score and pass state from normalized components."""
    ts = recompute_score_from_checkpoints(out)
    passed = ts >= pass_threshold * 100 - 1e-6
    return out.model_copy(update={"total_score_0_100": ts, "passed": passed})
