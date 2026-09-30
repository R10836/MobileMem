"""LLM Judge：按检查点评分标准输出结构化 JSON；支持多模型并行与均值汇总。"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from loguru import logger

from eval import config
from eval.core.capabilities import (
    _DEDUCTION_MAX_POINTS_RE,
    _deduction_capability_slots,
    deduction_rule_lines,
    max_points_from_score_rule,
    score_rule_lines,
)
from eval.judging.client import chat_completion_for_model, parse_json_loose
from eval.core.schemas import JudgeOutput, normalize_judge_output
from eval.core.tasks import TaskItem
from eval.traces.jsonl import ConversationTurn, _content_part_kind


JUDGE_SYSTEM = """你是手机 AI Agent 记忆系统的专业评测裁判。你的任务是严格按照给定的标准，客观评估助手回复的质量。

【输入信息】
- 评测 Prompt
- 期望行为
- 【GT参考内容】（若有）：事实锚，记忆检索类得分须对照 GT 条目
- 检查点评分标准（得分项）
- 若材料中含「【扣分标准】」章节，则为扣分项规则（误报、事实错误、语言不自然等）
- 助手实际回复
- 工具调用摘要

【表述要求（面向非专业读者）】
- checkpoints[].reason、deductions[].reason、rationale 一律用直白、好懂的中文，避免“评测腔”和黑话。
- 建议结构：先写结论（得几分 / 扣几分 / 不扣），再写 1～2 句关键依据；少用从句套从句。
- 满分得分原因可写「本项要求已做到：……」；未满分须说明差在哪，但不要和扣分项混写在同一句里。

【条数与字段一一对应（硬性）】
- checkpoints 条数必须**恰好等于**【检查点评分标准】中的得分细则条数（按 ①②③ 或独立行计）；deductions 条数必须**恰好等于**【扣分标准】中的扣分细则条数（同条细则含主档「扣分=」与「按 N/处」附加档时，按细则拆成多条 deductions，与用户消息中的条数说明一致）。
- **禁止遗漏**任一得分/扣分细则，**禁止凭空新增**检查点或扣分条；id 从 0 起递增，与细则顺序一致。
- 每条输出的 score / points 必须与 reason 叙述一致：reason 写「不扣分」则 points 必须为 0；reason 写「扣 N 分」则 points 必须等于 N；reason 写「得 M 分」则 score 必须等于 M（且 ≤ max_points）。**禁止**出现「按 2 分/条共扣 6 分」但 points=0，或「本条不扣分」但 points>0 等矛盾。

【评分规则】（与用例说明一致：得分项满分 100 分逐项加分；扣分项从总得分中扣除；最终得分 = 总得分 − 总扣分，不设下限，可为负数）
1. 逐项对照【检查点评分标准】中每个得分检查点评分，**一条细则对应 checkpoints 中恰好一条**。
2. 每个检查点输出：id、reason、max_points（与原文分值一致，若原文无分值则按条目均分使满分合计约 100）、score（0～max_points）、passed（score == max_points）。reason 为本检查点**得分侧**原因（每条必填，1～3 句大白话、可审计）；reason 中若写分值，必须与 score 一致。
3. 扣分项：仅当用户消息中提供了「【扣分标准】」时，**一条扣分细则（含多档时按档拆条）对应 deductions 中恰好一条**。输出 occurrences（触发次数）；「扣分=N/处」由系统按 N×occurrences 重算，只有原文明确写出“最多/上限”时才封顶。未触发必须 occurrences=0、points=0。**遗漏类问题只在 checkpoints 少得分体现，deductions 对应 points 必须为 0。**
4. 若未提供【扣分标准】，deductions 必须为空数组 []。
5. earned = 所有 checkpoints 的 score 之和；deducted = 所有 deductions 的 points 之和；total_score_0_100 = earned − deducted（不设下限，可为负；你输出的 total_score 字段须与该式一致）。
6. 判定 passed：total_score_0_100 >= 用户给定的百分制阈值。
7. rationale：1～3 句大白话草稿即可；系统会在校正后按 checkpoints/deductions 的**最终** score/points 重写，你须保证各条 reason 与 score/points 一致，且 total_score_0_100 = 得分合计 − 扣分合计（勿在 rationale 里写与算术不符的总分或「满分通过」）。

【记忆检索 · 得分检查点（能力=记忆检索 或细则含「逐条计分」「按 GT 关键条目」）】
- **只按细则给分**：细则写「按 GT 关键条目等比分配、每正确召回 1 条得对应分」时，须先数清 GT 中 ①②③… 条目，再逐条判断助手回复是否语义召回了该条关键事实，累加 score，且不超过 max_points。
- **得分原因必须写清「因召回了哪些 GT 而得分」**：逐条列出已召回的 GT 序号与简短概括（如「已召回 GT①日历：…；GT②文档：…」），并在 matched_gt_ids 中输出对应序号。不得只写笼统的「覆盖较好」。
- **未召回的 GT 条目**：可在同一条 reason 里用「未召回 GT③…」说明为何少得分，但**禁止**写「因漏召扣分」「误召导致扣分」「另扣××分」等扣分侧表述；得分原因里**不得出现「误召」二字**。
- **禁止满分（硬性）**：下列任一情况出现时，该条 score **必须小于** max_points（不得 passed=true），且 reason 与 score 须一致：
  - 混入非 GT、遗漏/未召回/部分召回/召回不全/与 GT 不符；
  - 承认 GT 要求的关键人名、机构、日期、金额等**未在助手回复中出现或未列出**（如「正文未列出」「未详细列出」「仅泛称科技大佬但未列出具体人物」）；
  - 用「虽然…但未…」承认某 GT 条目仅部分满足或回复中缺少该条关键事实。
- **禁止「叙述漏召却给满分」**：若 reason 写明关注人物/账单/日历等关键事实「未在正文列出」「未出现」「仅板块标题无具体内容」，不得同时将 score 评为 max_points；应按未召回或部分召回给分。
- 非记忆检索类检查点：按该条细则描述的能力要求评分，reason 同样用大白话说明满足/未满足点。

【分值上限（硬性）】
- 每条 checkpoint 的 max_points **必须等于**对应得分细则原文「分值=N」中的 N（用户消息中会给出对照表）。
- 每条 deduction 的 unit_points 必须等于对应细则的单次扣分 N；occurrences 是实际触发次数。
- 「扣分=N/处」若未声明总上限，points 可以超过 N；系统会按 unit_points×occurrences 重算。
- score 不得超过该条 max_points；固定扣分或明确有上限的扣分不得超过其上限。

【记忆检索 · 扣分检查点（扣分细则针对误召）】
- **只按误召扣分**：仅当助手回复出现事实错误、混入其他事件、证据外编造、金额/人物/时间错配等**误召**情形时，才按该条「扣分=…/处」及「逐条扣分细则」计 points。
- **扣分原因必须写清「因误召了哪些内容而扣分」**：写明回复中的错误表述，并对照 GT 说明错在哪（如「误召：把 GT② 的 6 月账单写成 5 月，扣 12 分」）。未触发时写「未发现符合本条的误召，不扣分」，points=0。
- **扣分原因禁止写漏召**：不得出现「漏召」「未召回 GT」「未提及」「未覆盖」「遗漏」等遗漏类表述；遗漏只在得分检查点体现。

【语义粒度（概括 vs 详述）】
- 助手回复可能是**高度概括**（只保留 GT 主旨、省略细节）或**较详细**（接近 GT 原文层次）。二者均可判为正确召回，只要核心事实与 GT 条目语义一致。
- 概括与详述均可得分：不要求回复与 GT 日期/金额字面一致；中文月日、省略年份、同义改写均可能算召回。
- **记忆检索类额外要求**：GT 若明确要求具体人物/机构/金额/日期，回复或 reason 中承认「未列出、未出现、仅泛称」时，只能按部分召回计分，**不得满分**。
- 你在 reason 中仍须自洽：若写「已召回 GT①」而回复明显未涉及该条主线，应少给分；**禁止**捏造回复中不存在且与 GT 无关的具体事实后仍给高分。

【输出格式】
只输出一个 JSON 对象，不要使用 Markdown 代码块或其他额外文本。
"""


@dataclass
class JudgeEnsembleResult:
    """多模型或单模型 Judge 的统一结果；汇总分与通过判定基于成功模型的平均分。"""

    mean_total_score_0_100: float | None
    aggregate_passed: bool | None
    report: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.mean_total_score_0_100 is not None


def _split_grading_criteria_sections(criteria: str) -> tuple[str, str]:
    s = str(criteria or "")
    m = re.search(r"【\s*扣分标准\s*】", s)
    if not m:
        return s, ""
    return s[: m.start()], s[m.end() :]


def _extract_rule_lines(section_text: str) -> list[str]:
    out: list[str] = []
    for raw in str(section_text or "").splitlines():
        t = raw.strip()
        if not t:
            continue
        if t.startswith("【") and t.endswith("】"):
            continue
        t = re.sub(r"^[\-\*\u2022]\s*", "", t)
        t = re.sub(r"^\(?\d+\)?[\.、:：]\s*", "", t)
        t = re.sub(r"^（\d+）\s*", "", t)
        t = t.strip()
        if t:
            out.append(t)
    return out


def _expected_rule_counts(task: TaskItem) -> tuple[int, int]:
    """得分细则条数；扣分条数含同条多档展开（与 grading_capability 口径一致）。"""
    criteria = task.grading_criteria
    score_n = len(score_rule_lines(criteria))
    if score_n <= 0:
        score_n = 1
    ded_n = 0
    for line in deduction_rule_lines(criteria):
        ded_n += _deduction_capability_slots(line)
    return score_n, ded_n


def _deduction_rule_per_slot(task: TaskItem) -> list[str]:
    """与 deductions 下标对齐：每条 slot 对应其来源扣分细则原文。"""
    out: list[str] = []
    for line in deduction_rule_lines(task.grading_criteria):
        for _ in range(_deduction_capability_slots(line)):
            out.append(line)
    return out


_PER_ITEM_DEDUCTION_MAX_RE = re.compile(r"按\s*(\d+(?:\.\d+)?)\s*/\s*处")
_PRIMARY_PER_OCCURRENCE_RE = re.compile(
    r"扣分\s*=\s*(\d+(?:\.\d+)?)\s*/\s*处"
)
_EXPLICIT_DEDUCTION_CAP_RE = re.compile(
    r"(?:最多扣|最高扣|扣分上限|上限)\s*[=:：]?\s*(\d+(?:\.\d+)?)\s*分?"
)


@dataclass(frozen=True)
class _DeductionRuleSpec:
    unit_points: float
    per_occurrence: bool
    cap_points: float | None = None


def _deduction_max_points_per_slot(line: str) -> list[float]:
    """单条扣分细则各档上限：主档「扣分=N」+ 同条内「按 M/处」附加档。"""
    s = str(line or "").strip()
    m = _DEDUCTION_MAX_POINTS_RE.search(s)
    if not m:
        return [0.0]
    vals = [float(m.group(1))]
    head = s.split("逐条扣分细则")[0]
    head = _DEDUCTION_MAX_POINTS_RE.sub("", head, count=1)
    for em in _PER_ITEM_DEDUCTION_MAX_RE.finditer(head):
        vals.append(float(em.group(1)))
    return vals


def _deduction_rule_specs(task: TaskItem) -> list[_DeductionRuleSpec]:
    specs: list[_DeductionRuleSpec] = []
    for line in deduction_rule_lines(task.grading_criteria):
        base = _DEDUCTION_MAX_POINTS_RE.search(line)
        cap_match = _EXPLICIT_DEDUCTION_CAP_RE.search(line)
        cap = float(cap_match.group(1)) if cap_match else None
        if base:
            unit = float(base.group(1))
            specs.append(
                _DeductionRuleSpec(
                    unit_points=unit,
                    per_occurrence=bool(_PRIMARY_PER_OCCURRENCE_RE.search(line)),
                    cap_points=cap,
                )
            )
            head = _DEDUCTION_MAX_POINTS_RE.sub("", line.split("逐条扣分细则")[0], count=1)
            for match in _PER_ITEM_DEDUCTION_MAX_RE.finditer(head):
                specs.append(
                    _DeductionRuleSpec(
                        unit_points=float(match.group(1)),
                        per_occurrence=True,
                        cap_points=cap,
                    )
                )
        else:
            specs.append(_DeductionRuleSpec(unit_points=0.0, per_occurrence=False))
    return specs


def _expected_checkpoint_max_points(task: TaskItem) -> list[float]:
    lines = score_rule_lines(task.grading_criteria)
    if not lines:
        return [100.0]
    mxs = [max_points_from_score_rule(ln) for ln in lines]
    if all(abs(x) <= 1e-9 for x in mxs):
        share = 100.0 / len(lines)
        return [share] * len(lines)
    return mxs


def _expected_deduction_max_points(task: TaskItem) -> list[float]:
    return [
        spec.cap_points if spec.cap_points is not None else spec.unit_points
        for spec in _deduction_rule_specs(task)
    ]


# reason 承认记忆检索不完美时不得满分（用短语/正则，避免「只是」「没有…不符」误伤）
_MEMORY_IMPERFECTION_NEG_PHRASES = (
    "未遗漏",
    "无遗漏",
    "未漏召",
    "没有遗漏",
    "未发现遗漏",
    "未混入",
    "没有缺失",
    "没有遗漏",
    "没有虚构",
    "没有误召",
    "没有编造",
    "没有发现",
    "并未遗漏",
    "并无遗漏",
    "未出现其他",
    "没有出现其他",
    "未出现与事实",
    "未出现与GT",
    "未出现与 GT",
    "未出现与其他",
    "没有出现与其他",
)
_MEMORY_IMPERFECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"混入",
        r"误并入",
        r"非\s*GT",
        r"未在\s*GT",
        r"不在\s*GT",
        r"与\s*GT\s*不符",
        r"未召回",
        r"漏召",
        r"未覆盖",
        r"未提及",
        r"未命中",
        r"遗漏",
        r"部分召回",
        r"部分覆盖",
        r"(?:召回|覆盖)不全",
        r"覆盖不足",
        r"严重不足",
        r"仅部分|仅有部分|仅覆盖",
        r"召回不足",
        r"召回较低",
        r"未完整(?:召回|覆盖)?",
        r"未准确(?:识别|召回|覆盖)?",
        r"未明确提及",
        r"主干召回不全",
        r"错配",
        r"召回准确率较低",
        r"明显偏离",
        r"正文未",
        r"未在(?:正文|回复|助手回复|最终回复)",
        r"未详细列出",
        r"未充分列出",
        r"没有列出",
        r"未列出",
        r"仅泛称",
        r"只(?:能)?泛泛",
        r"仅泛泛",
        r"泛泛提及",
        r"泛泛地说",
        r"泛泛而谈",
        r"未写出",
        r"未写明",
        r"虽然.{0,30}未(?:详细)?列出",
        r"虽未(?:详细)?列出",
        r"但未(?:在|于)(?:正文|回复)",
        r"但未列出",
        r"但未出现",
        r"但未提及",
        r"但未覆盖",
        r"无具体(?:人物|人名|名单|金额|日期)",
    )
)


def _reason_for_imperfection_scan(reason: str) -> str:
    r = str(reason or "")
    for neg in _MEMORY_IMPERFECTION_NEG_PHRASES:
        r = r.replace(neg, "")
    return r


def _reason_admits_memory_retrieval_imperfection(reason: str) -> bool:
    r = _reason_for_imperfection_scan(reason)
    return any(p.search(r) for p in _MEMORY_IMPERFECTION_PATTERNS)


def _criteria_max_points_block(task: TaskItem) -> str:
    cp_mxs = _expected_checkpoint_max_points(task)
    ded_specs = _deduction_rule_specs(task)
    cp_lines = "\n".join(f"  - 得分细则第 {i} 条 max_points={mx:g}" for i, mx in enumerate(cp_mxs))
    if ded_specs:
        ded_lines = "\n".join(
            (
                f"  - 扣分细则第 {i} 条 unit_points={spec.unit_points:g}, "
                f"按处={'是' if spec.per_occurrence else '否'}, "
                f"总上限={spec.cap_points:g}"
                if spec.cap_points is not None
                else f"  - 扣分细则第 {i} 条 unit_points={spec.unit_points:g}, "
                f"按处={'是' if spec.per_occurrence else '否'}, 无总上限"
            )
            for i, spec in enumerate(ded_specs)
        )
        ded_sec = f"\n{ded_lines}"
    else:
        ded_sec = "\n  - （本任务无扣分标准）"
    return f"""
【细则分值上限对照（输出必须与下列数值一致）】
得分检查点：
{cp_lines}
扣分条目：{ded_sec}
"""


def _sync_max_points_from_criteria(out: JudgeOutput, task: TaskItem) -> JudgeOutput:
    """Align scores and compute per-occurrence deductions deterministically."""
    exp_cp = _expected_checkpoint_max_points(task)
    ded_specs = _deduction_rule_specs(task)
    new_cps = []
    for i, cp in enumerate(out.checkpoints):
        mx = max(0.0, exp_cp[i] if i < len(exp_cp) else float(cp.max_points or 0))
        sc = max(0.0, min(float(cp.score or 0), mx))
        new_cps.append(
            cp.model_copy(
                update={
                    "max_points": mx,
                    "score": sc,
                    "passed": abs(sc - mx) <= 1e-6,
                }
            )
        )
    new_deds = []
    for i, d in enumerate(out.deductions):
        if i >= len(ded_specs):
            new_deds.append(d)
            continue
        spec = ded_specs[i]
        raw_points = max(0.0, float(d.points or 0))
        occurrences = max(0, int(d.occurrences or 0))
        if occurrences == 0 and raw_points > 1e-9:
            if spec.per_occurrence and spec.unit_points > 1e-9:
                occurrences = max(1, int(round(raw_points / spec.unit_points)))
            else:
                occurrences = 1
        if spec.per_occurrence:
            points = spec.unit_points * occurrences
        else:
            points = spec.unit_points if occurrences else 0.0
        if spec.cap_points is not None:
            points = min(points, spec.cap_points)
        new_deds.append(
            d.model_copy(
                update={
                    "unit_points": spec.unit_points,
                    "occurrences": occurrences,
                    "max_points": spec.cap_points,
                    "points": points,
                }
            )
        )
    return out.model_copy(update={"checkpoints": new_cps, "deductions": new_deds})


def _enforce_memory_retrieval_no_full_if_admitted(
    out: JudgeOutput, task: TaskItem
) -> JudgeOutput:
    """记忆检索得分：reason 承认混入/遗漏等时禁止 score==max_points。"""
    rules = _score_rule_lines(task)
    new_cps = []
    for i, cp in enumerate(out.checkpoints):
        rule = rules[i] if i < len(rules) else ""
        if not _rule_is_memory_retrieval(rule):
            new_cps.append(cp)
            continue
        mx = float(cp.max_points or 0)
        sc = float(cp.score or 0)
        rsn = str(cp.reason or "")
        if mx <= 1e-9 or sc < mx - 1e-6:
            new_cps.append(cp)
            continue
        if not _reason_admits_memory_retrieval_imperfection(rsn):
            new_cps.append(cp)
            continue
        cap = max(0.0, mx - max(1.0, mx * 0.05))
        if sc > cap + 1e-6:
            logger.debug(
                "记忆检索满分校正 task_id={} cp={} score {} -> {}（reason 承认不完美）",
                task.task_id,
                i,
                sc,
                cap,
            )
        new_cps.append(
            cp.model_copy(
                update={
                    "score": cap,
                    "passed": False,
                }
            )
        )
    return out.model_copy(update={"checkpoints": new_cps})


def _validate_output_rule_coverage(task: TaskItem, out: JudgeOutput) -> None:
    exp_score_n, exp_ded_n = _expected_rule_counts(task)
    got_score_n = len(out.checkpoints or [])
    got_ded_n = len(out.deductions or [])

    if got_score_n != exp_score_n:
        raise ValueError(
            f"checkpoints 条数须与得分细则一致：期望 {exp_score_n} 条，实际 {got_score_n} 条"
        )
    if exp_ded_n == 0:
        if got_ded_n != 0:
            raise ValueError(
                f"未提供扣分标准时 deductions 应为空，实际 {got_ded_n} 条"
            )
        return
    if got_ded_n != exp_ded_n:
        raise ValueError(
            f"deductions 条数须与扣分细则一致：期望 {exp_ded_n} 条，实际 {got_ded_n} 条"
        )


_OMISSION_DEDUCTION_PATTERNS = (
    "未召回",
    "漏召",
    "未提及",
    "未覆盖",
    "未命中",
    "缺失",
    "遗漏",
)

_MEMORY_RETRIEVAL_MARKERS = ("记忆检索", "能力=记忆检索")

# 得分原因中不应出现的「误召/漏召扣分」类表述（记忆检索得分项）
_FORBIDDEN_SCORE_SIDE_CLAUSE = re.compile(
    r"[；;，,]?[^；;，,。]*(?:误召|漏召|未覆盖|未提及|未召回)[^；;，,。]*(?:扣|扣分|另扣|重复扣)[^；;，,。]*",
    re.IGNORECASE,
)
_FORBIDDEN_SCORE_SIDE_PHRASE = re.compile(
    r"(?:因|由于)?(?:误召|漏召)[^。；;]{0,40}(?:扣|扣分)",
    re.IGNORECASE,
)

# 扣分原因中不应出现的遗漏类表述
_FORBIDDEN_OMISSION_IN_DEDUCTION = re.compile(
    r"[；;，,]?[^；;，,。]*(?:漏召|未召回|未提及|未覆盖|未命中|遗漏|缺失)[^；;，,。]*",
    re.IGNORECASE,
)

_DEDUCTION_NO_FALSE_RECALL_DEFAULT = "未发现符合本条的误召，本条不扣分。"
_DEDUCTION_OMISSION_IN_SCORE_ONLY = (
    "属于漏召或未覆盖，已在得分检查点体现，本条不扣分。"
)

# reason 中「已扣分」叙述（points=0 时应剔除）
_DEDUCTION_AMOUNT_CLAUSE_RE = re.compile(
    r"[；;，,]?[^；;，,。]*(?:"
    r"按\s*\d+(?:\.\d+)?\s*分\s*/?\s*(?:条|处)"
    r"|共扣\s*\d+(?:\.\d+)?"
    r"|累计[^。；]{0,24}扣\s*\d+(?:\.\d+)?"
    r"|扣\s*\d+(?:\.\d+)?\s*分"
    r"|\d+\s*条[^。；]{0,24}扣"
    r")[^；;，,。]*",
    re.IGNORECASE,
)
_CLAIMED_DEDUCTION_POINTS_RE = re.compile(
    r"(?:共|累计)[^。；]{0,12}扣\s*(\d+(?:\.\d+)?)|扣\s*(\d+(?:\.\d+)?)\s*分",
    re.IGNORECASE,
)
_CLAIMED_SCORE_POINTS_RE = re.compile(
    r"(?:实)?得\s*(\d+(?:\.\d+)?)\s*分",
    re.IGNORECASE,
)

# 层 1：可结构归一化的事实锚点（日期、金额）；不核对专名/地名/中英文别名
_DATE_IN_TEXT_RE = re.compile(
    r"\d{4}\s*[-/年]\s*\d{1,2}(?:\s*[-/月]\s*\d{1,2})?"
    r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?"
)
_MONEY_IN_TEXT_RE = re.compile(
    r"[¥￥]\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*元"
)
_POSITIVE_RECALL_CLAUSE_RE = re.compile(
    r"已召回|已识别|已覆盖|准确识别|正确识别|召回GT|召回 GT",
    re.IGNORECASE,
)
_REASON_AUDIT_NOTE_RE = re.compile(r"（核对[^）]*）")
_GT_CIRCLE_IDS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵"
_GT_ID_IN_TEXT_RE = re.compile(rf"GT\s*([{_GT_CIRCLE_IDS}])", re.IGNORECASE)
_GT_ENTRY_HEAD_RE = re.compile(rf"^([{_GT_CIRCLE_IDS}])")

def _parse_date_to_key(raw: str) -> str | None:
    s = str(raw or "").strip()
    m = re.match(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]?\s*(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}{mo:02d}{d:02d}"
    m = re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}{mo:02d}{d:02d}"
    return None


def _parse_money_to_key(raw: str) -> str | None:
    s = str(raw or "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return f"{float(m.group(1)):.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return None


def _date_keys_from_text(text: str) -> set[str]:
    keys: set[str] = set()
    for m in _DATE_IN_TEXT_RE.finditer(str(text or "")):
        k = _parse_date_to_key(m.group(0))
        if k:
            keys.add(k)
    return keys


def _money_keys_from_text(text: str) -> set[str]:
    keys: set[str] = set()
    for m in _MONEY_IN_TEXT_RE.finditer(str(text or "")):
        raw = m.group(0)
        k = _parse_money_to_key(raw)
        if k:
            keys.add(k)
    return keys


def _structural_anchor_count(text: str) -> int:
    return len(_date_keys_from_text(text)) + len(_money_keys_from_text(text))


def _structural_support_ratio(source_text: str, assistant_text: str) -> tuple[float, int]:
    """
    层 1：source 中的日期/金额键是否在 assistant 中出现（归一化后集合交集）。
    返回 (支持率, 锚点数量)；锚点数为 0 时支持率视为 1.0（无硬事实可核对）。
    """
    d_src = _date_keys_from_text(source_text)
    m_src = _money_keys_from_text(source_text)
    n = len(d_src) + len(m_src)
    if n == 0:
        return 1.0, 0
    d_ans = _date_keys_from_text(assistant_text)
    m_ans = _money_keys_from_text(assistant_text)
    hits = len(d_src & d_ans) + len(m_src & m_ans)
    return hits / n, n


def _parse_gt_entries(metadata_gt: str) -> dict[str, str]:
    """解析 metadata.GT 为 {「①」: 条目正文, ...}。"""
    raw = str(metadata_gt or "").strip()
    if not raw:
        return {}
    parts = re.split(rf"(?=[{_GT_CIRCLE_IDS}])", raw)
    out: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _GT_ENTRY_HEAD_RE.match(part)
        if not m:
            continue
        gid = m.group(1)
        body = part[len(gid) :].strip()
        if body.startswith(":") or body.startswith("："):
            body = body[1:].strip()
        out[gid] = body
    return out


def _normalize_gt_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^GT\s*", "", text, flags=re.IGNORECASE)
    match = re.search(rf"[{_GT_CIRCLE_IDS}]", text)
    return match.group(0) if match else ""


def _enforce_memory_gt_equal_weight(out: JudgeOutput, task: TaskItem) -> JudgeOutput:
    """Recompute memory checkpoint scores from the matched GT labels."""
    entries = _parse_gt_entries(task.metadata_gt or "")
    if not entries:
        return out
    valid_ids = set(entries)
    rules = _score_rule_lines(task)
    checkpoints = []
    for index, checkpoint in enumerate(out.checkpoints):
        rule = rules[index] if index < len(rules) else ""
        if not _rule_is_memory_retrieval(rule):
            checkpoints.append(checkpoint)
            continue
        # ``matched_gt_ids`` is the machine-readable scoring contract.  Do
        # not recover it from prose: an omitted field must not earn memory
        # points merely because the free-form reason happens to mention GT.
        raw_ids = list(checkpoint.matched_gt_ids or [])
        matched: list[str] = []
        for raw_id in raw_ids:
            gt_id = _normalize_gt_id(raw_id)
            if gt_id in valid_ids and gt_id not in matched:
                matched.append(gt_id)
        max_points = float(checkpoint.max_points or 0)
        score = max_points * len(matched) / len(entries)
        checkpoints.append(
            checkpoint.model_copy(
                update={
                    "matched_gt_ids": matched,
                    "score": score,
                    "passed": len(matched) == len(entries),
                }
            )
        )
    return out.model_copy(update={"checkpoints": checkpoints})


def _claimed_recalled_gt_ids(reason: str) -> list[str]:
    """从 reason 正向召回叙述中解析 Judge 声称已召回的 GT 序号（去重保序）。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for seg in re.split(r"[；;。]", str(reason or "")):
        if not _POSITIVE_RECALL_CLAUSE_RE.search(seg):
            continue
        for m in _GT_ID_IN_TEXT_RE.finditer(seg):
            gid = m.group(1)
            neg = re.search(
                rf"未(?:召回|覆盖|提及|命中)\s*GT\s*{re.escape(gid)}",
                seg,
                re.IGNORECASE,
            )
            if neg:
                continue
            if gid in seen:
                continue
            seen.add(gid)
            ordered.append(gid)
    return ordered


def _layer2_gt_claim_support_ratio(
    reason: str, metadata_gt: str, assistant_text: str
) -> tuple[float, int, int]:
    """
    层 2：对 reason 中声称已召回的每条 GT，用该条 GT 原文中的日期/金额锚点核对回复。
    返回 (平均支持率, 参与核对的 GT 条数, 总硬锚点数)。
    """
    entries = _parse_gt_entries(metadata_gt)
    claimed = _claimed_recalled_gt_ids(reason)
    if not entries or not claimed:
        return 1.0, 0, 0
    ratios: list[float] = []
    anchor_total = 0
    for gid in claimed:
        body = entries.get(gid, "")
        if not body:
            continue
        ratio, n = _structural_support_ratio(body, assistant_text)
        if n == 0:
            continue
        anchor_total += n
        ratios.append(ratio)
    if not ratios:
        return 1.0, 0, anchor_total
    return sum(ratios) / len(ratios), len(ratios), anchor_total


def _gt_claimed_structural_anchor_count(reason: str, metadata_gt: str) -> int:
    entries = _parse_gt_entries(metadata_gt)
    total = 0
    for gid in _claimed_recalled_gt_ids(reason):
        total += _structural_anchor_count(entries.get(gid, ""))
    return total


def _resolve_verify_ratio(
    *,
    memory: bool,
    reason: str,
    metadata_gt: str,
    assistant_text: str,
) -> tuple[float, str]:
    """
    记忆检索且存在 GT：优先层 2；否则层 1（reason 内硬事实）。
    返回 (ratio, mode_label)。
    """
    if memory and str(metadata_gt or "").strip():
        ratio, n_gt, n_anch = _layer2_gt_claim_support_ratio(
            reason, metadata_gt, assistant_text
        )
        if n_gt > 0 and n_anch > 0:
            return ratio, "gt_entry"
        if n_gt > 0 and n_anch == 0:
            return 1.0, "gt_entry_no_structural"
    r1, n1 = _structural_support_ratio(
        _positive_recall_text(reason), assistant_text
    )
    if n1 > 0:
        return r1, "reason_structural"
    return 1.0, "skip"


def _positive_recall_text(reason: str) -> str:
    chunks: list[str] = []
    for seg in re.split(r"[；;。]", str(reason or "")):
        if _POSITIVE_RECALL_CLAUSE_RE.search(seg):
            chunks.append(seg)
    return "；".join(chunks)


def _extract_structural_tokens_from_reason(reason: str) -> list[str]:
    """测试/调试：从 reason 正向叙述中提取日期/金额字面量。"""
    text = _positive_recall_text(reason)
    tokens: list[str] = []
    for m in _DATE_IN_TEXT_RE.finditer(text):
        tokens.append(m.group(0))
    for m in _MONEY_IN_TEXT_RE.finditer(text):
        tokens.append(m.group(0))
    return tokens


def _recall_claim_support_ratio(reason: str, assistant_text: str) -> float:
    """测试/调试：对 reason 正向叙述中的硬事实锚点计算支持率。"""
    src = _positive_recall_text(reason)
    ratio, _ = _structural_support_ratio(src, assistant_text)
    return ratio


def _reason_claims_positive_recall(reason: str) -> bool:
    return bool(_POSITIVE_RECALL_CLAUSE_RE.search(str(reason or "")))


def _strip_verify_audit_notes(out: JudgeOutput) -> JudgeOutput:
    """移除历史版本在 reason 末尾追加的「核对…」压分备注。"""
    new_cps = []
    for cp in out.checkpoints:
        rsn = _REASON_AUDIT_NOTE_RE.sub("", str(cp.reason or "")).strip()
        new_cps.append(cp.model_copy(update={"reason": rsn}))
    return out.model_copy(update={"checkpoints": new_cps})


def _enforce_reason_supported_by_answer(
    out: JudgeOutput,
    task: TaskItem,
    assistant_text: str,
) -> JudgeOutput:
    """
    保留入口以兼容调用链；不再根据层 1/层 2 日期金额字面匹配压分。

    纯字面核对无法覆盖中文月日、同义概括与 Judge 已给出的 GT 条数得分，
    易与模型语义评判冲突，故仅由 Judge 在打分阶段自洽，后处理不改 score。
    """
    _ = (task, assistant_text)
    return _strip_verify_audit_notes(out)


def _sync_rationale_from_scores(
    out: JudgeOutput, *, pass_threshold: float
) -> JudgeOutput:
    """按校正后的 checkpoints/deductions 重写 rationale，避免与总分/通过态矛盾。"""
    earned = sum(float(c.score or 0) for c in out.checkpoints)
    deducted = sum(float(d.points or 0) for d in out.deductions)
    ts = float(out.total_score_0_100)
    thr = pass_threshold * 100

    earn_bits: list[str] = []
    for i, c in enumerate(out.checkpoints):
        sc = float(c.score or 0)
        mx = float(c.max_points or 0)
        if sc <= 1e-9:
            continue
        if mx > 1e-9 and abs(sc - mx) <= 1e-6:
            earn_bits.append(f"第{i + 1}项满分{sc:g}分")
        else:
            earn_bits.append(f"第{i + 1}项得{sc:g}分")

    ded_bits = [
        f"扣{d.points:g}分"
        for d in out.deductions
        if float(d.points or 0) > 1e-9
    ]

    lines = [
        "得分方面："
        + ("，".join(earn_bits) if earn_bits else "各检查点基本不得分"),
        "扣分方面：" + ("，".join(ded_bits) if ded_bits else "无误召扣分"),
        (
            f"最终合计 {ts:g} 分（得分 {earned:g} − 扣分 {deducted:g}），"
            f"{'通过' if out.passed else '未通过'}（通过线 {thr:g} 分）"
        ),
    ]
    return out.model_copy(update={"rationale": "；".join(lines) + "。"})


def _score_rule_lines(task: TaskItem) -> list[str]:
    return score_rule_lines(task.grading_criteria)


def _deduction_rule_lines(task: TaskItem) -> list[str]:
    return deduction_rule_lines(task.grading_criteria)


def _reason_claims_positive_deduction(reason: str) -> bool:
    r = str(reason or "")
    if _DEDUCTION_AMOUNT_CLAUSE_RE.search(r):
        return True
    if _CLAIMED_DEDUCTION_POINTS_RE.search(r):
        return True
    if re.search(r"\d+\s*条[^。；]{0,24}扣", r):
        return True
    return False


def _strip_positive_deduction_narrative(reason: str) -> str:
    r = str(reason or "").strip()
    prev = None
    while prev != r:
        prev = r
        r = _DEDUCTION_AMOUNT_CLAUSE_RE.sub("", r)
        r = re.sub(r"扣\s*\d+(?:\.\d+)?\s*分", "", r, flags=re.IGNORECASE)
    r = re.sub(r"[；;，,]{2,}", "；", r)
    return re.sub(r"^[；;，,\s]+|[；;，,\s]+$", "", r).strip()


def _finalize_zero_point_deduction_reason(reason: str) -> str:
    """
    points=0 时润色扣分原因：只去掉与分值矛盾的「扣 N 分/累计扣分」叙述，保留可审计说明。

    不一律改成「未发现误召」——漏召、未触发、已检查无误等情形应保留或改用更准确模板。
    """
    r = str(reason or "").strip()
    if _reason_claims_positive_deduction(r):
        r = _strip_positive_deduction_narrative(r)
    r = re.sub(r"[；;，,]{2,}", "；", r)
    r = re.sub(r"^[；;，,\s]+|[；;，,\s]+$", "", r).strip()
    if not r:
        return _DEDUCTION_NO_FALSE_RECALL_DEFAULT
    if _is_omission_like_reason(r):
        if any(k in r for k in ("得分", "体现", "不重复", "已在", "检查点")):
            return r if "不扣分" in r else f"{r}（本条不扣分）"
        return _DEDUCTION_OMISSION_IN_SCORE_ONLY
    if any(k in r for k in ("未触发", "未发现", "不扣分", "未扣分", "本条不扣")):
        return r
    if any(k in r for k in ("误召", "事实错误", "错配", "混入", "编造")):
        if "不扣分" in r or "未达" in r:
            return r
        return f"{r}（本条未达扣分标准，不扣分）"
    return r if "不扣分" in r else f"{r}（本条不扣分）"


def _align_deduction_reason(reason: str, *, points: float) -> str:
    """扣分原因与 points 对齐，避免「写了扣 6 分但 points=0」。"""
    pts = float(points or 0)
    r = str(reason or "").strip()
    if abs(pts) <= 1e-9:
        return _finalize_zero_point_deduction_reason(r)
    if any(
        k in r
        for k in ("不扣分", "未扣分", "未发现符合", "未发现误召", "未触发", "本条不扣")
    ) and not _reason_claims_positive_deduction(r):
        return f"存在误召，本条实扣 {pts:g} 分。"
    claimed = _claimed_deduction_points_from_reason(r)
    if claimed is not None and abs(claimed - pts) > 1e-6:
        r = _strip_positive_deduction_narrative(r)
        if not r or _reason_claims_positive_deduction(r):
            return f"存在误召，本条实扣 {pts:g} 分。"
        return f"{r}（本条实扣 {pts:g} 分）"
    if not _reason_claims_positive_deduction(r):
        return f"{r}（本条实扣 {pts:g} 分）" if r else f"存在误召，本条实扣 {pts:g} 分。"
    return r


def _claimed_deduction_points_from_reason(reason: str) -> float | None:
    m = _CLAIMED_DEDUCTION_POINTS_RE.search(str(reason or ""))
    if not m:
        return None
    g = m.group(1) or m.group(2)
    try:
        return float(g)
    except (TypeError, ValueError):
        return None


def _align_checkpoint_reason(reason: str, *, score: float, max_points: float) -> str:
    """得分原因与 score 对齐。"""
    r = str(reason or "").strip()
    if not r:
        return r
    full = abs(score - max_points) <= 1e-6
    m = _CLAIMED_SCORE_POINTS_RE.search(r)
    if m:
        claimed = float(m.group(1))
        target = max_points if full else score
        if abs(claimed - target) > 1e-6:
            r = _CLAIMED_SCORE_POINTS_RE.sub(f"实得 {target:g} 分", r, count=1)
    if full and re.search(r"(?:未得|不得满分|只得|尚有不足)", r):
        r = re.sub(
            r"[；;，,]?[^；;，,。]*(?:未得|不得满分|只得|尚有不足)[^；;，,。]*",
            "",
            r,
        ).strip()
    if not full and re.search(r"(?:满分|得满|全部做到|已全部做到)", r):
        r = re.sub(
            r"[；;，,]?[^；;，,。]*(?:满分|得满|全部做到|已全部做到)[^；;，,。]*",
            "",
            r,
        ).strip()
    return r.strip() or (
        "本检查点要求已全部做到。" if full else "本检查点还有要求未做到。"
    )


def _rule_is_memory_retrieval(rule_line: str) -> bool:
    s = str(rule_line or "")
    return any(m in s for m in _MEMORY_RETRIEVAL_MARKERS)


def _is_omission_like_reason(reason: str) -> bool:
    r = str(reason or "").strip()
    if not r:
        return False
    return any(k in r for k in _OMISSION_DEDUCTION_PATTERNS)


def _sanitize_memory_retrieval_score_reason(reason: str) -> str:
    """记忆检索得分原因：去掉误召/漏召扣分侧表述，保留已召回/未召回 GT 叙述。"""
    r = str(reason or "").strip()
    if not r:
        return r
    prev = None
    while prev != r:
        prev = r
        r = _FORBIDDEN_SCORE_SIDE_CLAUSE.sub("", r)
        r = _FORBIDDEN_SCORE_SIDE_PHRASE.sub("", r)
    r = re.sub(r"[；;，,]{2,}", "；", r)
    r = re.sub(r"^[；;，,\s]+|[；;，,\s]+$", "", r).strip()
    return r or "已按细则核对 GT 条目召回情况。"


def _sanitize_memory_retrieval_deduction_reason(reason: str, *, points: float) -> str:
    """记忆检索扣分原因：只保留误召侧说明；遗漏类表述剔除或替换为未触发模板。"""
    r = str(reason or "").strip()
    pts = float(points or 0)
    if abs(pts) <= 1e-9:
        return _align_deduction_reason(r, points=0)
    prev = None
    while prev != r:
        prev = r
        r = _FORBIDDEN_OMISSION_IN_DEDUCTION.sub("", r)
    r = re.sub(r"[；;，,]{2,}", "；", r)
    r = re.sub(r"^[；;，,\s]+|[；;，,\s]+$", "", r).strip()
    if not r or _is_omission_like_reason(r):
        return _align_deduction_reason("存在误召情形，已按细则扣分。", points=pts)
    return _align_deduction_reason(r, points=pts)


def _polish_judge_reasons(out: JudgeOutput, task: TaskItem) -> JudgeOutput:
    """按能力子项润色 reason 文案，强化得分/扣分原因分工及与分值一致。"""
    score_rules = _score_rule_lines(task)
    ded_rules = _deduction_rule_per_slot(task)

    new_cps = []
    for i, cp in enumerate(out.checkpoints):
        rsn = str(cp.reason or "").strip()
        score = float(cp.score or 0)
        max_p = float(cp.max_points or 0)
        rule = score_rules[i] if i < len(score_rules) else ""
        if _rule_is_memory_retrieval(rule):
            rsn = _sanitize_memory_retrieval_score_reason(rsn)
        rsn = _align_checkpoint_reason(rsn, score=score, max_points=max_p)
        new_cps.append(cp.model_copy(update={"reason": rsn}))

    new_deds = []
    for i, d in enumerate(out.deductions):
        rsn = str(d.reason or "").strip()
        pts = float(d.points or 0)
        rule = ded_rules[i] if i < len(ded_rules) else ""
        if _rule_is_memory_retrieval(rule):
            rsn = _sanitize_memory_retrieval_deduction_reason(rsn, points=pts)
        elif abs(pts) <= 1e-9 and (not rsn or _is_omission_like_reason(rsn)):
            rsn = _align_deduction_reason("", points=0)
        else:
            rsn = _align_deduction_reason(rsn, points=pts)
        new_deds.append(d.model_copy(update={"reason": rsn}))

    rationale = str(out.rationale or "").strip()
    return out.model_copy(update={"checkpoints": new_cps, "deductions": new_deds, "rationale": rationale})


def _grading_eval_guides_block(task: TaskItem) -> str:
    exp_score_n, exp_ded_n = _expected_rule_counts(task)
    ded_note = (
        f"须输出 deductions **{exp_ded_n}** 条（与扣分细则条数一致，含多档拆条）"
        if exp_ded_n
        else "不得输出 deductions（本任务无扣分标准）"
    )
    return f"""
【评测书写要求（务必遵守）】
1. 得分原因、扣分原因、rationale 均用小白也能读懂的大白话，避免「评测腔」。
2. **条数硬性要求**：checkpoints 必须 **{exp_score_n}** 条，与得分细则一一对应；{ded_note}。禁止遗漏、禁止多写。
3. **原因与分值一致**：reason 里写的「得 M 分」必须等于该条 score；写的「扣 N 分」必须等于该条 points。扣分须填写 occurrences；points=0 时 occurrences=0，且禁止写累计扣分明细。
4. 记忆检索类**得分**细则：reason 须写明「已召回 GT①…」，matched_gt_ids 须列出相同序号；**禁止**在得分原因里写误召/漏召扣分。
5. 记忆检索类**扣分**细则：仅因误召扣分；须写明误召内容；**禁止**写漏召；未触发时 points=0 且 reason 不得出现扣分明细。
6. 记忆检索类**得分**：reason 若承认混入/遗漏/部分召回/与 GT 不符，或承认关键事实「未在正文/回复列出、未详细列出、仅泛称无具体内容」，则 score 必须小于 max_points（不得满分）；叙述与 score 不得矛盾。
7. 每条 max_points 必须与【细则分值上限对照】一致，不得自行改写分值上限。
8. 助手回复可能是高度概括或较详细表述：概括可得分，但若 GT 要求的具体人物/金额等未在回复出现，须在 reason 中按未召回/部分召回少给分，不得满分。
9. **记忆检索自洽**：不得出现「已召回 GT②人物清单，但正文未列出 Musk/Altman/黄仁勋」且 score 仍为满分；要么少写已召回，要么降低 score。
"""


def _suppress_omission_double_penalty(out: JudgeOutput) -> JudgeOutput:
    """
    避免“漏召/未覆盖”在 checkpoints 失分后又在 deductions 重复扣分。
    保留原因文本用于审计，但将这类 deduction 的 points 归零。
    """
    changed = False
    patched = []
    for d in out.deductions:
        reason = str(d.reason or "").strip()
        pts = float(d.points or 0)
        if (pts > 0 or int(d.occurrences or 0) > 0) and _is_omission_like_reason(reason):
            changed = True
            patched.append(d.model_copy(update={"points": 0.0, "occurrences": 0}))
        else:
            patched.append(d)
    if not changed:
        return out
    return out.model_copy(update={"deductions": patched})


JUDGE_MAX_ATTEMPTS = 3

def _final_reply_max_chars() -> int:
    return config.judge_final_reply_max_chars()


def _thinking_max_chars() -> int:
    return config.judge_thinking_max_chars()


def _tool_summary_max_chars() -> int:
    return config.judge_tool_summary_max_chars()


def _clip_text_for_payload(text: str, max_chars: int) -> tuple[str, bool, int]:
    raw = str(text or "")
    raw_len = len(raw)
    if raw_len <= max_chars:
        return raw, False, raw_len
    clipped = raw[:max_chars]
    note = f"\n\n[内容已截断：原始 {raw_len} 字符，仅保留前 {max_chars} 字符]"
    return clipped + note, True, raw_len


def _assistant_thinking_text(turn: ConversationTurn) -> str:
    texts: list[str] = []
    for block in turn.assistant_blocks:
        msg = block.get("message", block)
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content") or []:
            if not isinstance(part, dict):
                continue
            if _content_part_kind(part) != "thinking":
                continue
            th = part.get("thinking") or part.get("text") or ""
            s = str(th).strip()
            if s:
                texts.append(s)
    return "\n\n".join(texts).strip()


def _tool_summary(
    turn: ConversationTurn, max_len: int | None = None
) -> tuple[str, bool, int]:
    if max_len is None:
        max_len = _tool_summary_max_chars()
    lines: list[str] = []
    for i, ev in enumerate(turn.iter_tool_call_events(), start=1):
        args = json.dumps(ev.arguments, ensure_ascii=False)[:500]
        tid = ev.tool_call_id or "(no-id)"
        lines.append(f"{i}. {ev.name} id={tid} {args}")
        if ev.thinking_before_call:
            tb = ev.thinking_before_call[:800].replace("\n", " ")
            lines.append(f"   thinking_before: {tb!r}")
        r = ev.result
        if r is None:
            lines.append("   result: <无配对 toolResult>")
        else:
            snippet = r.raw_content[:400].replace("\n", " ")
            lines.append(f"   result error={r.is_error} {snippet!r}")
    text = "\n".join(lines)
    clipped, truncated, raw_len = _clip_text_for_payload(text, max_len)
    return clipped, truncated, raw_len


def _build_user_payload(
    task: TaskItem,
    turn: ConversationTurn,
    *,
    pass_threshold: float,
) -> str:
    assistant_text_raw = turn.final_assistant_text()
    assistant_thinking_raw = _assistant_thinking_text(turn)
    assistant_text, final_truncated, final_raw_len = _clip_text_for_payload(
        assistant_text_raw, _final_reply_max_chars()
    )
    assistant_thinking, thinking_truncated, thinking_raw_len = _clip_text_for_payload(
        assistant_thinking_raw, _thinking_max_chars()
    )
    tools, tool_truncated, tool_raw_len = _tool_summary(turn)
    gt_ref = str(task.metadata_gt or "").strip()
    expected = str(task.expected_behavior or "").strip()
    gt_section = ""
    if gt_ref:
        gt_body = gt_ref
        if expected and expected != gt_ref:
            gt_body = gt_ref + "\n（补充）expected_behavior:\n" + expected
        gt_section = f"""
【GT参考内容（来自任务 metadata.GT）】
{gt_body}
"""
    exp_score_n, exp_ded_n = _expected_rule_counts(task)
    return f"""
【任务说明】
请根据以下标准对【助手最终回复】进行客观评分，输出 JSON 格式结果。
{_grading_eval_guides_block(task)}
{_criteria_max_points_block(task)}
【通过阈值】
最终得分（总得分 − 总扣分，可为负）需 >= {pass_threshold * 100:.1f} 分为通过。

【评测 Prompt】
{task.prompt}

【期望行为 Expected Behavior】
{task.expected_behavior}
{gt_section}

【检查点评分标准 / 扣分标准 Grading Criteria】
{task.grading_criteria}

【助手思考内容（用于评分参考，不等同最终回复）】
{assistant_thinking or "(无显式思考内容)"}

【助手最终回复（面向用户）】
{assistant_text or "(无文本回复)"}

【工具调用摘要】
{tools or "(无工具调用)"}

【内容截断说明】
- 思考内容：{"已截断" if thinking_truncated else "未截断"}（原始 {thinking_raw_len} 字符）
- 最终回复：{"已截断" if final_truncated else "未截断"}（原始 {final_raw_len} 字符）
- 工具摘要：{"已截断" if tool_truncated else "未截断"}（原始 {tool_raw_len} 字符）

【输出格式】
请严格输出以下 JSON 格式，不要添加任何额外文本。
- checkpoints 数组长度必须等于 {exp_score_n}（得分细则条数）
- deductions 数组长度必须等于 {exp_ded_n}（扣分细则条数，无扣分标准时为 0）

{{
  "checkpoints": [
    {{
      "id": "检查点唯一标识（从0开始整数递增，条数={exp_score_n}）",
      "reason": "本检查点得分原因（大白话；所写分值须与 score 一致；记忆检索须列 GT①②③）",
      "max_points": 满分分值,
      "score": 实际得分（数字，0 到 max_points 之间）,
      "passed": true（得分 == 满分） 或 false（得分 < 满分）,
      "matched_gt_ids": ["①", "②"]（仅记忆检索填写实际正确覆盖的 GT 序号，其他能力写 []）
    }}
  ],
  "deductions": [
    {{
      "reason": "本条扣分原因（大白话；所写扣分数须与 points 一致；points=0 时不得写累计扣分）",
      "points": 本条实际扣分（数字；未触发写 0）,
      "unit_points": 单次扣分值,
      "occurrences": 实际触发次数（未触发写 0）,
      "max_points": 原文明确给出的总上限；没有总上限写 null
    }}
  ],
  "total_score_0_100": 最终得分（数字，等于各 checkpoint score 之和减去各 deductions.points 之和；不设下限，可为负）,
  "passed": true（total_score_0_100 >= 阈值） 或 false,
  "rationale": "评分理由（1-3 句大白话：得了什么分、误召扣了什么、是否通过）"
}}

现在请输出评测结果。
"""


def _run_judge_ensemble(
    task: TaskItem,
    user_payload: str,
    *,
    pass_threshold: float,
    assistant_text: str = "",
) -> JudgeEnsembleResult:
    models = config.judge_model_names()
    per_model_rows: list[dict[str, Any]] = []
    successes: list[JudgeOutput] = []
    failure_msgs: list[str] = []
    max_workers = min(len(models), config.judge_parallel_max_workers())

    def run_m(m: str) -> tuple[str, JudgeOutput | None, str | None]:
        return _judge_one_model(
            task,
            user_payload,
            model=m,
            pass_threshold=pass_threshold,
            assistant_text=assistant_text,
        )

    if len(models) == 1:
        model, out, err = run_m(models[0])
        if out is not None:
            per_model_rows.append(
                {
                    "model": model,
                    "ok": True,
                    "total_score_0_100": out.total_score_0_100,
                    "passed": out.passed,
                    "judge": judge_output_to_dict(out),
                    "error": None,
                }
            )
            successes.append(out)
        else:
            per_model_rows.append(
                {
                    "model": model,
                    "ok": False,
                    "total_score_0_100": None,
                    "passed": None,
                    "judge": None,
                    "error": err,
                }
            )
            if err:
                failure_msgs.append(f"{model}: {err}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_m, m): m for m in models}
            for fut in as_completed(futs):
                model, out, err = fut.result()
                if out is not None:
                    per_model_rows.append(
                        {
                            "model": model,
                            "ok": True,
                            "total_score_0_100": out.total_score_0_100,
                            "passed": out.passed,
                            "judge": judge_output_to_dict(out),
                            "error": None,
                        }
                    )
                    successes.append(out)
                else:
                    per_model_rows.append(
                        {
                            "model": model,
                            "ok": False,
                            "total_score_0_100": None,
                            "passed": None,
                            "judge": None,
                            "error": err,
                        }
                    )
                    if err:
                        failure_msgs.append(f"{model}: {err}")
        per_model_rows.sort(key=lambda r: models.index(r["model"]) if r["model"] in models else 999)

    if not successes:
        report = {
            "mode": "multi" if len(models) > 1 else "single",
            "judge_models": models,
            "aggregate": {
                "mean_total_score_0_100": None,
                "aggregate_passed": None,
                "models_succeeded": 0,
                "models_total": len(models),
            },
            "per_model": per_model_rows,
            "failure_errors": failure_msgs,
        }
        return JudgeEnsembleResult(None, None, report)

    scores = [float(o.total_score_0_100) for o in successes]
    mean_score = sum(scores) / len(scores)
    aggregate_passed = mean_score >= pass_threshold * 100 - 1e-6

    report = {
        "mode": "multi" if len(models) > 1 else "single",
        "judge_models": models,
        "aggregate": {
            "mean_total_score_0_100": round(mean_score, 4),
            "aggregate_passed": aggregate_passed,
            "models_succeeded": len(successes),
            "models_total": len(models),
            "per_model_scores": scores,
        },
        "per_model": per_model_rows,
    }
    if failure_msgs:
        report["partial_failures"] = failure_msgs

    logger.debug(
        "Judge 汇总 task_id={} models_ok={}/{} mean_score={} aggregate_passed={}",
        task.task_id,
        len(successes),
        len(models),
        round(mean_score, 4),
        aggregate_passed,
    )

    return JudgeEnsembleResult(
        round(mean_score, 4),
        aggregate_passed,
        report,
    )


def _finalize_single_output(
    out: JudgeOutput,
    pass_threshold: float,
    *,
    task: TaskItem | None = None,
    task_id: str = "",
    assistant_text: str = "",
) -> JudgeOutput:
    """将模型给出的 total_score_0_100 校正为「总得分 − 总扣分」，并同步 passed。"""
    raw = float(out.total_score_0_100)
    no_double = _suppress_omission_double_penalty(out)
    synced = _sync_max_points_from_criteria(no_double, task) if task is not None else no_double
    equal_weighted = (
        _enforce_memory_gt_equal_weight(synced, task)
        if task is not None
        else synced
    )
    verified = (
        _enforce_reason_supported_by_answer(equal_weighted, task, assistant_text)
        if task is not None and assistant_text.strip()
        else equal_weighted
    )
    polished = _polish_judge_reasons(verified, task) if task is not None else verified
    fixed = normalize_judge_output(polished, pass_threshold=pass_threshold)
    synced_rationale = _sync_rationale_from_scores(
        fixed, pass_threshold=pass_threshold
    )
    ts = float(synced_rationale.total_score_0_100)
    if abs(raw - ts) > 1e-4:
        logger.debug(
            "Judge total_score_0_100 校正 task_id={} 模型输出={} -> 总得分-总扣分={}",
            task_id or "?",
            raw,
            ts,
        )
    return synced_rationale


def _judge_one_model(
    task: TaskItem,
    user_payload: str,
    *,
    model: str,
    pass_threshold: float,
    assistant_text: str = "",
) -> tuple[str, JudgeOutput | None, str | None]:
    last_err: str | None = None
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            logger.debug(
                "Judge 请求 task_id={} model={} attempt={}/{} pass_threshold={} prompt_len={}",
                task.task_id,
                model,
                attempt,
                JUDGE_MAX_ATTEMPTS,
                pass_threshold,
                len(user_payload),
            )
            raw = chat_completion_for_model(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_payload},
                ],
                model=model,
                temperature=config.judge_temperature(),
                response_format_json=True,
            )
            data = parse_json_loose(raw)
            out = _finalize_single_output(
                _coerce_judge_output(data),
                pass_threshold,
                task=task,
                task_id=task.task_id,
                assistant_text=assistant_text,
            )
            _validate_output_rule_coverage(task, out)
            logger.debug(
                "Judge 响应 task_id={} model={} attempt={}/{} total_score_0_100={} passed={}",
                task.task_id,
                model,
                attempt,
                JUDGE_MAX_ATTEMPTS,
                out.total_score_0_100,
                out.passed,
            )
            return model, out, None
        except Exception as e:
            last_err = str(e)
            logger.warning(
                "Judge 模型失败 task_id={} model={} attempt={}/{} err={}",
                task.task_id,
                model,
                attempt,
                JUDGE_MAX_ATTEMPTS,
                e,
            )
    return model, None, last_err


def run_judge(
    task: TaskItem,
    turn: ConversationTurn,
    *,
    pass_threshold: float,
) -> JudgeEnsembleResult:
    """
    按 config.judge_model_names() 列出的模型并行调用 Judge。
    成功返回的模型分数取算术平均作为 total_score_0_100 汇总；
    aggregate_passed = (mean >= pass_threshold * 100)。
    单模型时行为与原先一致，report 结构仍含 aggregate / per_model 便于解析。
    """
    user_payload = _build_user_payload(task, turn, pass_threshold=pass_threshold)
    return _run_judge_ensemble(
        task,
        user_payload,
        pass_threshold=pass_threshold,
        assistant_text=turn.final_assistant_text(),
    )


def _coerce_judge_output(data: dict[str, Any]) -> JudgeOutput:
    cps = []
    for i, c in enumerate(data.get("checkpoints") or []):
        if not isinstance(c, dict):
            continue
        max_p = float(c.get("max_points", 0) or 0)
        score = float(c.get("score", 0) or 0)
        rsn = str(c.get("reason", "") or c.get("description", "") or c.get("deduction_reason", "") or "").strip()
        if not rsn:
            rsn = (
                "本检查点要求已全部做到。"
                if abs(score - max_p) <= 1e-6
                else "本检查点还有要求未做到。"
            )
        cps.append(
            {
                "id": str(c.get("id", i)),
                "reason": rsn,
                "max_points": max_p,
                "passed": bool(c.get("passed", False)),
                "score": score,
                "matched_gt_ids": [
                    str(x).strip()
                    for x in (c.get("matched_gt_ids") or [])
                    if str(x).strip()
                ],
            }
        )
    deds = []
    for d in data.get("deductions") or []:
        if not isinstance(d, dict):
            continue
        pts = float(d.get("points", 0) or 0)
        m = d.get("max_points")
        try:
            m_f = float(m) if m is not None else None
        except (TypeError, ValueError):
            m_f = None
        rsn = str(d.get("reason", "") or "").strip()
        if not rsn:
            rsn = (
                _DEDUCTION_NO_FALSE_RECALL_DEFAULT
                if abs(pts) <= 1e-9
                else "存在符合本条的误召，已按细则扣分。"
            )
        deds.append(
            {
                "reason": rsn,
                "points": pts,
                "max_points": m_f,
                "unit_points": (
                    float(d.get("unit_points"))
                    if d.get("unit_points") is not None
                    else None
                ),
                "occurrences": int(d.get("occurrences", 0) or 0),
            }
        )
    raw_total = data.get("total_score_0_100", data.get("total_score", 0))
    return JudgeOutput.model_validate(
        {
            "checkpoints": cps,
            "deductions": deds,
            "total_score_0_100": float(raw_total or 0),
            "passed": bool(data.get("passed", False)),
            "rationale": str(data.get("rationale", "")),
        }
    )


def judge_output_to_dict(out: JudgeOutput) -> dict[str, Any]:
    return out.model_dump()
