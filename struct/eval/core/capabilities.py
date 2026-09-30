"""Parse capability items and aggregate Judge checkpoints and deductions."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

_CAPABILITY_RE = re.compile(r"能力\s*=\s*([^；;\n]+)")
_SCORE_MAX_POINTS_RE = re.compile(r"分值\s*=\s*(\d+(?:\.\d+)?)")
_DEDUCTION_MAX_POINTS_RE = re.compile(r"扣分\s*=\s*(\d+(?:\.\d+)?)")
# Match an additional per-item tier in the same rule, excluding its primary
# tier and any explanatory detail that follows the per-item section.
_DEDUCTION_PER_ITEM_RE = re.compile(r"按\s*\d+(?:\.\d+)?\s*/\s*处")
_DEDUCTION_PER_ITEM_VALUE_RE = re.compile(r"按\s*(\d+(?:\.\d+)?)\s*/\s*处")
_EXPLICIT_DEDUCTION_CAP_RE = re.compile(
    r"(?:最多扣|最高扣|扣分上限|上限)\s*[=:：]?\s*(\d+(?:\.\d+)?)\s*分?"
)

_UNLABELED_CAPABILITY = "未标注能力"
# Merge retrieval and reasoning/aggregation into one display capability.
_DISPLAY_RETRIEVAL_QA_CAPABILITY = "检索问答"
_MERGE_TO_RETRIEVAL_QA_SOURCE = frozenset({"记忆检索", "推理/聚合"})


def normalize_capability_name(capability: str) -> str:
    """Normalize whitespace and slash variants for matching and aggregation."""
    s = str(capability or "").strip().replace("／", "/")
    return re.sub(r"\s+", "", s)


_MERGE_TO_RETRIEVAL_QA_NORMALIZED = frozenset(
    normalize_capability_name(c) for c in _MERGE_TO_RETRIEVAL_QA_SOURCE
)


def display_capability(capability: str) -> str:
    """Return the display name used in capability summaries."""
    if normalize_capability_name(capability) in _MERGE_TO_RETRIEVAL_QA_NORMALIZED:
        return _DISPLAY_RETRIEVAL_QA_CAPABILITY
    return str(capability or "").strip() or _UNLABELED_CAPABILITY


def capability_from_rule_line(line: str) -> str:
    m = _CAPABILITY_RE.search(str(line or ""))
    return m.group(1).strip() if m else "未标注能力"


def _split_grading_criteria_sections(criteria: str) -> tuple[str, str]:
    s = str(criteria or "")
    m = re.search(r"【\s*扣分标准\s*】", s)
    if not m:
        return s, ""
    return s[: m.start()], s[m.end() :]


_CIRCLED_SPLIT = re.compile(
    r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵])"
)


def split_rule_line_fragments(line: str) -> list[str]:
    """Split one rule line at circled labels.

    A circled label starts a new rule only when its fragment contains a
    capability declaration. Otherwise it is treated as a step reference in
    the preceding rule.
    """
    line = str(line or "").strip()
    if not line:
        return []
    parts = [p.strip() for p in _CIRCLED_SPLIT.split(line) if p.strip()]
    if len(parts) <= 1:
        norm = _normalize_rule_chunk(line)
        return [norm] if norm else []

    merged: list[str] = []
    for p in parts:
        if _CAPABILITY_RE.search(p):
            merged.append(p)
        elif merged:
            merged[-1] = merged[-1] + p
        else:
            norm = _normalize_rule_chunk(p)
            if norm:
                merged.append(norm)
    return [x for x in (_normalize_rule_chunk(m) for m in merged) if x]


def _normalize_rule_chunk(chunk: str) -> str:
    t = chunk.strip()
    if not t:
        return ""
    if t.startswith("【") and t.endswith("】"):
        return ""
    t = re.sub(r"^[\-\*\u2022]\s*", "", t)
    t = re.sub(r"^\(?\d+\)?[\.、:：]\s*", "", t)
    t = re.sub(r"^（\d+）\s*", "", t)
    return t.strip()


def _extract_rule_lines(section_text: str) -> list[str]:
    """Extract individual rules from score or deduction sections."""
    s = str(section_text or "").strip()
    if not s:
        return []
    out: list[str] = []
    lines = [raw.strip() for raw in s.splitlines() if raw.strip()]
    if not lines:
        lines = [s]
    for line in lines:
        if line.startswith("【") and line.endswith("】"):
            continue
        out.extend(split_rule_line_fragments(line))
    if not out and s:
        out.extend(split_rule_line_fragments(s))
    return out


def score_rule_lines(grading_criteria: str) -> List[str]:
    score_part, _ = _split_grading_criteria_sections(grading_criteria)
    return _extract_rule_lines(score_part)


def deduction_rule_lines(grading_criteria: str) -> List[str]:
    _, ded_part = _split_grading_criteria_sections(grading_criteria)
    return _extract_rule_lines(ded_part)


def score_rule_capabilities(grading_criteria: str) -> List[str]:
    return [capability_from_rule_line(line) for line in score_rule_lines(grading_criteria)]


def _deduction_capability_slots(line: str) -> int:
    """Return the number of Judge deductions represented by one rule."""
    if not str(line or "").strip():
        return 0
    if not _DEDUCTION_MAX_POINTS_RE.search(line):
        return 1
    head = str(line).split("逐条扣分细则")[0]
    head = _DEDUCTION_MAX_POINTS_RE.sub("", head, count=1)
    extra = len(_DEDUCTION_PER_ITEM_RE.findall(head))
    return 1 + extra


def _collapse_consecutive_capabilities(caps: List[str]) -> List[str]:
    """Collapse consecutive duplicate capability slots."""
    out: List[str] = []
    for cap in caps:
        if not out or cap != out[-1]:
            out.append(cap)
    return out


def deduction_rule_capabilities(grading_criteria: str) -> List[str]:
    caps: List[str] = []
    for line in deduction_rule_lines(grading_criteria):
        cap = capability_from_rule_line(line)
        for _ in range(_deduction_capability_slots(line)):
            caps.append(cap)
    return caps


def deduction_capabilities_for_judge_items(
    grading_criteria: str,
    deductions: List[Dict[str, Any]],
) -> List[str]:
    """Align capability labels with the Judge deduction list."""
    expanded = deduction_rule_capabilities(grading_criteria)
    n = len(deductions)
    if n == len(expanded):
        return expanded
    collapsed = _collapse_consecutive_capabilities(expanded)
    if n == len(collapsed):
        return collapsed
    if n > len(collapsed):
        return expanded
    return collapsed[:n] if n < len(collapsed) else expanded


def max_points_from_score_rule(line: str) -> float:
    m = _SCORE_MAX_POINTS_RE.search(str(line or ""))
    if m:
        return float(m.group(1))
    return 0.0


def max_points_from_deduction_rule(line: str) -> float:
    m = _DEDUCTION_MAX_POINTS_RE.search(str(line or ""))
    if m:
        return float(m.group(1))
    return 0.0


def synthetic_zero_capability_judge_items(
    grading_criteria: str,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Create zero-valued Judge items when no LLM details are available."""
    checkpoints: List[Dict[str, float]] = []
    for line in score_rule_lines(grading_criteria):
        mx = max_points_from_score_rule(line)
        checkpoints.append({"score": 0.0, "max_points": mx})
    deductions: List[Dict[str, Any]] = []
    for line in deduction_rule_lines(grading_criteria):
        base = max_points_from_deduction_rule(line)
        head = _DEDUCTION_MAX_POINTS_RE.sub(
            "", str(line).split("逐条扣分细则")[0], count=1
        )
        units = [base] + [
            float(match.group(1))
            for match in _DEDUCTION_PER_ITEM_VALUE_RE.finditer(head)
        ]
        cap_match = _EXPLICIT_DEDUCTION_CAP_RE.search(line)
        cap = float(cap_match.group(1)) if cap_match else None
        for unit in units:
            deductions.append(
                {
                    "points": 0.0,
                    "max_points": cap,
                    "unit_points": unit,
                    "occurrences": 0,
                }
            )
    return checkpoints, deductions


def _is_no_effective_answer_row(row: Dict[str, Any]) -> bool:
    if row.get("match_status") == "no_effective_answer":
        return True
    if row.get("agent_effective_answer") is False:
        return True
    jb = row.get("judge")
    if isinstance(jb, dict) and (
        jb.get("mode") == "rule" or jb.get("skip_reason") == "no_effective_answer"
    ):
        return True
    return False


def judge_items_for_capability_stats(
    report_task: Dict[str, Any],
    judge: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return Judge items used for capability-level aggregation."""
    criteria = str(report_task.get("grading_criteria") or "")
    syn_cps, syn_deds = synthetic_zero_capability_judge_items(criteria)

    if _is_no_effective_answer_row(report_task) and (syn_cps or syn_deds):
        return (
            [dict(x) for x in syn_cps],
            [dict(x) for x in syn_deds],
        )

    j = judge or {}
    cps = [c for c in (j.get("checkpoints") or []) if isinstance(c, dict)]
    deds = [d for d in (j.get("deductions") or []) if isinstance(d, dict)]
    if cps or deds:
        return cps, deds
    if syn_cps or syn_deds:
        return [dict(x) for x in syn_cps], [dict(x) for x in syn_deds]
    return [], []


def primary_judge_dict(task_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    jb = task_row.get("judge")
    if not isinstance(jb, dict):
        return None
    for pm in jb.get("per_model") or []:
        if not isinstance(pm, dict):
            continue
        if pm.get("ok") and isinstance(pm.get("judge"), dict):
            return pm["judge"]
    return None


def _is_evaluated_task_row(row: Dict[str, Any]) -> bool:
    if row.get("skipped"):
        return False
    if primary_judge_dict(row) is not None:
        return True
    jb = row.get("judge")
    return (
        isinstance(jb, dict)
        and jb.get("mode") == "rule"
        and row.get("total_score_0_100") is not None
    )


def _per_task_capability_score_deduction(
    checkpoints: List[Dict[str, Any]],
    deductions: List[Dict[str, Any]],
    criteria: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Aggregate earned, maximum, and deducted points by capability."""
    score_caps = score_rule_capabilities(criteria)
    ded_caps = deduction_capabilities_for_judge_items(criteria, deductions)
    earned: DefaultDict[str, float] = defaultdict(float)
    max_score: DefaultDict[str, float] = defaultdict(float)
    deducted: DefaultDict[str, float] = defaultdict(float)

    for i, c in enumerate(checkpoints):
        cap = score_caps[i] if i < len(score_caps) else "未标注能力"
        earned[cap] += float(c.get("score", 0) or 0)
        max_score[cap] += float(c.get("max_points", 0) or 0)

    for i, d in enumerate(deductions):
        cap = ded_caps[i] if i < len(ded_caps) else "未标注能力"
        pts = float(d.get("points", 0) or 0)
        mx_raw = d.get("max_points")
        try:
            mx = float(mx_raw) if mx_raw is not None else pts
        except (TypeError, ValueError):
            mx = pts
        deducted[cap] += pts

    return dict(earned), dict(max_score), dict(deducted)


class CapabilityItemStatsAccumulator:
    """Accumulate checkpoint and deduction statistics by capability."""

    def __init__(self) -> None:
        self._score_sum: DefaultDict[str, float] = defaultdict(float)
        self._score_max_sum: DefaultDict[str, float] = defaultdict(float)
        self._score_n: DefaultDict[str, int] = defaultdict(int)
        self._rule_score_n: DefaultDict[str, int] = defaultdict(int)
        self._ded_sum: DefaultDict[str, float] = defaultdict(float)
        self._ded_capped_points_sum: DefaultDict[str, float] = defaultdict(float)
        self._ded_capped_max_sum: DefaultDict[str, float] = defaultdict(float)
        self._ded_n: DefaultDict[str, int] = defaultdict(int)
        self._ded_capped_n: DefaultDict[str, int] = defaultdict(int)
        self._ded_uncapped_n: DefaultDict[str, int] = defaultdict(int)
        self._rule_ded_n: DefaultDict[str, int] = defaultdict(int)
        self._all_capabilities: set[str] = set()
        # Per-case net score, maximum score, and involved capabilities.
        self._per_task_composite: List[
            Tuple[Dict[str, float], Dict[str, float], set[str]]
        ] = []

    def feed_task(
        self,
        report_task: Dict[str, Any],
        checkpoints: List[Dict[str, Any]],
        deductions: List[Dict[str, Any]],
    ) -> None:
        criteria = str(report_task.get("grading_criteria") or "")
        score_caps = score_rule_capabilities(criteria)
        ded_caps = deduction_capabilities_for_judge_items(criteria, deductions)

        for i, c in enumerate(checkpoints):
            cap = score_caps[i] if i < len(score_caps) else _UNLABELED_CAPABILITY
            disp = display_capability(cap)
            sc = float(c.get("score", 0) or 0)
            mx = float(c.get("max_points", 0) or 0)
            self._score_sum[disp] += sc
            self._score_max_sum[disp] += mx
            self._score_n[disp] += 1

        for i, d in enumerate(deductions):
            cap = ded_caps[i] if i < len(ded_caps) else _UNLABELED_CAPABILITY
            disp = display_capability(cap)
            pts = float(d.get("points", 0) or 0)
            mx_raw = d.get("max_points")
            self._ded_sum[disp] += pts
            self._ded_n[disp] += 1
            if mx_raw is None:
                self._ded_uncapped_n[disp] += 1
            else:
                try:
                    mx = float(mx_raw)
                except (TypeError, ValueError):
                    self._ded_uncapped_n[disp] += 1
                else:
                    self._ded_capped_points_sum[disp] += pts
                    self._ded_capped_max_sum[disp] += mx
                    self._ded_capped_n[disp] += 1

        earned, max_score, deducted = _per_task_capability_score_deduction(
            checkpoints, deductions, criteria
        )
        raw_involved = set(score_caps) | set(ded_caps)
        net_by_cap: Dict[str, float] = defaultdict(float)
        max_by_cap: Dict[str, float] = defaultdict(float)
        involved_display: set[str] = set()
        for cap in raw_involved:
            disp = display_capability(cap)
            involved_display.add(disp)
            net_by_cap[disp] += earned.get(cap, 0.0) - deducted.get(cap, 0.0)
            max_by_cap[disp] += max_score.get(cap, 0.0)
        self._all_capabilities.update(involved_display)
        self._per_task_composite.append(
            (dict(net_by_cap), dict(max_by_cap), involved_display)
        )

    def feed_row_if_evaluated(self, report_task: Dict[str, Any]) -> None:
        if not _is_evaluated_task_row(report_task):
            return
        criteria = str(report_task.get("grading_criteria") or "")
        for cap in score_rule_capabilities(criteria):
            self._rule_score_n[display_capability(cap)] += 1
        for cap in deduction_rule_capabilities(criteria):
            self._rule_ded_n[display_capability(cap)] += 1
        j = primary_judge_dict(report_task)
        cps, deds = judge_items_for_capability_stats(report_task, j)
        self.feed_task(report_task, cps, deds)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_by_capability": _finalize_score_buckets(
                self._score_sum,
                self._score_max_sum,
                self._score_n,
                self._rule_score_n,
            ),
            "deduction_by_capability": _finalize_deduction_buckets(
                self._ded_sum,
                self._ded_capped_points_sum,
                self._ded_capped_max_sum,
                self._ded_n,
                self._ded_capped_n,
                self._ded_uncapped_n,
                self._rule_ded_n,
            ),
            "composite_by_capability": _finalize_composite_buckets(
                self._all_capabilities,
                self._per_task_composite,
            ),
            "capability_subitem_count": len(self._all_capabilities),
        }


def _finalize_score_buckets(
    sum_score: Dict[str, float],
    sum_max: Dict[str, float],
    mean_counts: Dict[str, int],
    rule_counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cap, n_mean in mean_counts.items():
        if n_mean <= 0:
            continue
        ms = sum_score[cap] / n_mean
        mm = sum_max[cap] / n_mean
        out[cap] = {
            "mean_score": round(ms, 4),
            "mean_max_points": round(mm, 4),
            "mean_rate": round(ms / mm, 4) if mm > 1e-9 else None,
            "rule_occurrence_count": rule_counts.get(cap, 0),
            "item_count": rule_counts.get(cap, 0),
        }
    return out


def _finalize_deduction_buckets(
    sum_pts: Dict[str, float],
    capped_sum_pts: Dict[str, float],
    capped_sum_max: Dict[str, float],
    mean_counts: Dict[str, int],
    capped_counts: Dict[str, int],
    uncapped_counts: Dict[str, int],
    rule_counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cap, n_mean in mean_counts.items():
        if n_mean <= 0:
            continue
        mp = sum_pts[cap] / n_mean
        capped_n = capped_counts.get(cap, 0)
        capped_mp = capped_sum_pts[cap] / capped_n if capped_n else None
        capped_mm = capped_sum_max[cap] / capped_n if capped_n else None
        out[cap] = {
            "mean_points": round(mp, 4),
            "mean_max_points": round(capped_mm, 4) if capped_mm is not None else None,
            "mean_rate": (
                round(capped_sum_pts[cap] / capped_sum_max[cap], 4)
                if capped_sum_max[cap] > 1e-9
                else None
            ),
            "mean_capped_points": (
                round(capped_mp, 4) if capped_mp is not None else None
            ),
            "capped_item_count": capped_n,
            "uncapped_item_count": uncapped_counts.get(cap, 0),
            "rule_occurrence_count": rule_counts.get(cap, 0),
            "item_count": rule_counts.get(cap, 0),
        }
    return out


def _finalize_composite_buckets(
    all_capabilities: set[str],
    per_task: List[Tuple[Dict[str, float], Dict[str, float], set[str]]],
) -> Dict[str, Dict[str, Any]]:
    rows = _iter_composite_capability_rows(all_capabilities, per_task)
    out: Dict[str, Dict[str, Any]] = {}
    for cap, net, mx, rate, n_involved in rows:
        out[cap] = {
            "mean_net_score": round(net, 4),
            "mean_max_points": round(mx, 4),
            "mean_rate": round(rate, 4) if mx > 1e-9 else None,
            "involved_case_count": n_involved,
        }
    return out


def _iter_composite_capability_rows(
    all_capabilities: set[str],
    per_task: List[Tuple[Dict[str, float], Dict[str, float], set[str]]],
) -> List[Tuple[str, float, float, float, int]]:
    if not all_capabilities or not per_task:
        return []
    rows: List[Tuple[str, float, float, float, int]] = []
    for cap in sorted(all_capabilities):
        net_sum = 0.0
        max_sum = 0.0
        n_involved = 0
        for net_by_cap, max_by_cap, involved_caps in per_task:
            if cap not in involved_caps:
                continue
            n_involved += 1
            net_sum += net_by_cap.get(cap, 0.0)
            max_sum += max_by_cap.get(cap, 0.0)
        if n_involved <= 0:
            continue
        mean_net = net_sum / n_involved
        mean_max = max_sum / n_involved
        rate = mean_net / mean_max if mean_max > 1e-9 else 0.0
        rows.append((cap, mean_net, mean_max, rate, n_involved))
    return rows


def summarize_capability_item_stats(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc = CapabilityItemStatsAccumulator()
    for row in per_task:
        acc.feed_row_if_evaluated(row)
    return acc.to_summary_dict()
