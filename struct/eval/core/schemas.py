"""Judge 输出与评测报告结构。"""

from __future__ import annotations

from typing import Any, List, Optional

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
    """最终得分 = 各检查点 score 之和 − 各扣分 points 之和；不设下限（可为负），不截断到 0～100。"""
    earned = sum(c.score for c in out.checkpoints)
    deduct = sum(d.points for d in out.deductions)
    return earned - deduct


def normalize_judge_output(out: JudgeOutput, *, pass_threshold: float) -> JudgeOutput:
    """校正大模型输出：强制 total_score_0_100 = 总得分 − 总扣分，并据阈值重算 passed。"""
    ts = recompute_score_from_checkpoints(out)
    passed = ts >= pass_threshold * 100 - 1e-6
    return out.model_copy(update={"total_score_0_100": ts, "passed": passed})
