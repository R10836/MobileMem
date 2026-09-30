"""单条 / 批量 / dry-run 编排。"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from eval.judging.evaluator import run_judge
from eval.traces.jsonl import ConversationTurn, ParsedSession
from eval import config
from eval.core.tasks import (
    TaskItem,
    load_tasks_document,
    match_task_to_session,
)
from eval.metrics.tool_efficiency import compute_tool_efficiency
from eval.metrics.tool_effectiveness import compute_tool_use_effectiveness
from eval.metrics.answer_information import compute_answer_information_metrics
from eval.evidence.recall import (
    compute_evidence_recall,
    resolve_benchmark_data_root,
    resolve_index_user_id,
)
from eval.evidence.catalog import (
    EvidencePreflightError,
    is_active_evidence_root,
    load_active_evidence_index,
    validate_tasks_against_index,
)
from eval.traces.resolver import (
    load_parsed_session_resolved,
    resolve_log_target_prefer_bases,
)
from eval.core.capabilities import primary_judge_dict


def _format_batch_task_progress(
    index: int,
    total: int,
    task_id: str,
    row: Dict[str, Any],
) -> str:
    if row.get("skipped") is True:
        errors = row.get("errors") or []
        detail = str(errors[0]) if errors else str(row.get("match_status") or "未知原因")
        if len(detail) > 60:
            detail = detail[:57] + "..."
        return f"[{index}/{total}] {task_id} 跳过 | {detail}"
    score = row.get("total_score_0_100")
    if score is None:
        return f"[{index}/{total}] {task_id} 完成"
    result = "通过" if row.get("passed") is True else "未通过"
    return f"[{index}/{total}] {task_id} 完成 | {float(score):.2f} 分 | {result}"


def _print_batch_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _log_bases_for_tasks_file(
    tasks_path: Path, loaded_format: str, log_base_dir: Optional[Path]
) -> List[Path]:
    del loaded_format
    if log_base_dir is not None:
        return [log_base_dir.resolve()]
    return [tasks_path.resolve().parent, Path.cwd()]


def _session_getter_for_bases(bases: List[Path]):
    session_cache: Dict[str, ParsedSession] = {}

    def resolve_spec(spec: str) -> Path:
        p, _ = resolve_log_target_prefer_bases(bases, spec)
        return p

    def get_session(spec: str) -> ParsedSession:
        p = resolve_spec(spec)
        key = str(p)
        if key not in session_cache:
            sess = load_parsed_session_resolved(p)
            session_cache[key] = sess
            n_all = len(sess.turns)
            n_biz = len(sess.business_turns())
            logger.info(
                "日志解析完成 path={} spec={!r} session_id={} 全量交互轮次={} 业务向交互轮次={} 来源jsonl文件数={}",
                p,
                spec,
                sess.session_id,
                n_all,
                n_biz,
                len(sess.source_paths or []),
            )
        else:
            logger.debug("复用已加载会话: {}", key)
        return session_cache[key]

    return get_session, resolve_spec


def prompt_eval_fingerprint(prompt: str) -> str:
    """用例 Query 指纹（仅 prompt 文本，不含 task_id）。"""
    text = str(prompt or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def task_eval_fingerprint(task: TaskItem) -> str:
    return prompt_eval_fingerprint(task.prompt or "")


def row_eval_fingerprint(row: Dict[str, Any]) -> str:
    """检查点行 Query 指纹，与 task_eval_fingerprint 相同（仅 prompt）。"""
    fp = prompt_eval_fingerprint(str(row.get("prompt") or ""))
    if fp:
        return fp
    stored = row.get("task_eval_fingerprint")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    return ""


def row_eval_content_matches_task(row: Dict[str, Any], task: TaskItem) -> bool:
    if not row_eval_fingerprint(row):
        return False
    return row_eval_fingerprint(row) == task_eval_fingerprint(task)


def _merge_checkpoint_row_with_task(row: Dict[str, Any], task: TaskItem) -> Dict[str, Any]:
    """保留 Judge/匹配等评测结果，同步当前任务元数据（含 task_id）。"""
    out = copy.deepcopy(row)
    out.update(task_fields_public(task))
    return out


def task_fields_public(task: TaskItem) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "task_id": task.task_id,
        "name": task.name,
        "capability": task.capability,
        "sub_capability": task.sub_capability,
        "dimension": task.dimension,
        "query_type": task.query_type,
        "data_sources": task.data_sources,
        "prompt": task.prompt,
        "expected_behavior": task.expected_behavior,
        "grading_criteria": task.grading_criteria,
        "task_eval_fingerprint": task_eval_fingerprint(task),
    }
    gt = str(task.metadata_gt or "").strip()
    if gt:
        row["metadata_gt"] = gt
    return row


def _resolve_log_spec(
    doc_log: Optional[str],
    log_override: Optional[str],
    tasks_path: Path,
    *,
    loaded_format: str,
) -> str:
    del loaded_format
    spec = log_override or doc_log
    if not spec:
        raise ValueError(f"No agent trace was provided for {tasks_path}; use --log")
    return spec


def _active_evidence_preflight(
    tasks: List[TaskItem],
    evidence_root: Optional[Path],
    *,
    strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate cases once before matching traces or invoking a Judge."""
    if evidence_root is None:
        return None
    root = Path(evidence_root).expanduser().resolve()
    if not is_active_evidence_root(root):
        return None
    index = load_active_evidence_index(root)
    report = validate_tasks_against_index(tasks, index, evidence_root=root)
    if strict and not report.ok:
        raise EvidencePreflightError(report)
    return report.as_dict()


def _active_evidence_root_text(root: Optional[Path]) -> Optional[str]:
    """Return a report-friendly active evidence root."""
    if root is None:
        return None
    resolved = Path(root).expanduser().resolve()
    return str(resolved) if is_active_evidence_root(resolved) else None


def run_dry_run(
    tasks_path: Union[str, Path],
    log_override: Optional[str],
    *,
    log_base_dir: Optional[Path] = None,
    with_evidence_recall: bool = True,
    benchmark_data_root: Optional[Path] = None,
    evidence_fuzzy: bool = False,
    evidence_fuzzy_threshold: float = 0.86,
    with_answer_information_metrics: bool = False,
) -> Dict[str, Any]:
    tasks_path = Path(tasks_path)
    loaded = load_tasks_document(tasks_path)
    logger.info(
        "已加载评测任务条数={} (任务文件格式={})",
        len(loaded.tasks),
        loaded.format,
    )
    er_root_dr = benchmark_data_root if benchmark_data_root is not None else resolve_benchmark_data_root()
    # Evidence integrity is independent of whether the optional recall metric
    # is enabled. Dry-run reports errors; formal commands use strict mode.
    preflight = _active_evidence_preflight(
        loaded.tasks, er_root_dr, strict=False
    )
    idx_uid_dr = resolve_index_user_id(tasks_path)
    logger.info(
        "证据召回配置: benchmark_data_root={} index_user_id={}",
        str(er_root_dr.resolve()) if er_root_dr else None,
        idx_uid_dr,
    )
    bases = _log_bases_for_tasks_file(tasks_path, loaded.format, log_base_dir)
    default_spec = _resolve_log_spec(
        loaded.log_path, log_override, tasks_path, loaded_format=loaded.format
    )
    get_session, resolve_spec = _session_getter_for_bases(bases)

    default_target = resolve_spec(default_spec)
    default_session = get_session(default_spec)
    logger.info(
        "dry-run 默认日志 spec={!r} -> {} 解析基目录={} | 默认会话汇总: 全量交互轮次={} 业务向交互轮次={}",
        default_spec,
        default_target,
        [str(b) for b in bases],
        len(default_session.turns),
        len(default_session.business_turns()),
    )
    logger.info(
        "dry-run 指标开关: evidence_recall={} answer_information_metrics={}",
        with_evidence_recall,
        with_answer_information_metrics,
    )

    rows = []
    for task in loaded.tasks:
        spec = task.session_log or default_spec
        target = resolve_spec(spec)
        session = get_session(spec)
        m = match_task_to_session(task, session)
        logger.debug(
            "dry-run 匹配 task_id={} status={} biz_turn_idx={}",
            task.task_id,
            m.status,
            m.business_turn_index,
        )
        business = session.business_turns()
        rows.append(
            {
                "task_id": m.task.task_id,
                "status": m.status,
                "log_spec": spec,
                "resolved_log_target": str(target),
                "source_files": session.source_paths or [],
                "matched_turn_source_log": m.turn.source_log_path if m.turn else None,
                "session_id": session.session_id,
                "business_turn_count": len(business),
                "match_global_turn_index": m.global_turn_index,
                "match_business_turn_index": m.business_turn_index,
                "match_turn_index": m.business_turn_index,
                "prompt_preview": m.task.prompt[:120] + ("…" if len(m.task.prompt) > 120 else ""),
                "user_turn_preview": (m.turn.user_normalized[:120] + "…")
                if m.turn and len(m.turn.user_normalized) > 120
                else (m.turn.user_normalized if m.turn else None),
                "agent_final_answer": m.turn.final_assistant_text() if m.turn else None,
                "agent_total_tokens": m.turn.agent_total_tokens() if m.turn else None,
                "agent_processing_seconds": m.turn.agent_processing_duration_seconds()
                if m.turn
                else None,
                "warnings": m.warnings,
            }
        )
        er_root = benchmark_data_root if benchmark_data_root is not None else resolve_benchmark_data_root()
        _inject_evidence_recall(
            rows[-1],
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=er_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            rows[-1],
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
            dry_run=True,
        )
    out: Dict[str, Any] = {
        "tasks_path": str(tasks_path.resolve()),
        "task_file_format": loaded.format,
        "evidence_root": _active_evidence_root_text(er_root_dr),
        "default_log_spec": default_spec,
        "default_log_target": str(default_target),
        "default_source_files": default_session.source_paths or [],
        "default_log_path": str(default_target),
        "with_answer_information_metrics": with_answer_information_metrics,
        "matches": rows,
    }
    if loaded.load_stats is not None:
        out["case_load_stats"] = loaded.load_stats
    if loaded.skipped_empty_prompt:
        out["skipped_empty_prompt"] = loaded.skipped_empty_prompt
    if preflight is not None:
        out["evidence_preflight"] = preflight
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_skip = sum(1 for r in rows if r["status"] == "skipped")
    n_err = sum(1 for r in rows if r["status"] == "error")
    logger.info(
        "dry-run 完成: 任务数={} 匹配成功={} 跳过={} 错误={}",
        len(rows),
        n_ok,
        n_skip,
        n_err,
    )
    return out


def _extended_metrics_log_fragment(base: Dict[str, Any]) -> str:
    """工具有效性 / 证据召回 / 答案信息点 单行摘要（供 DEBUG 详细日志）。"""
    chunks: List[str] = []
    tue = base.get("tool_use_effectiveness")
    if tue is None:
        chunks.append("工具有效性=未计算")
    elif isinstance(tue, dict):
        err = tue.get("error")
        if err:
            err_s = str(err).replace("\n", " ")
            chunks.append(f"工具有效性=失败({err_s[:120]}{'…' if len(err_s) > 120 else ''})")
        else:
            ratio = tue.get("effective_tool_ratio")
            tot = tue.get("tool_call_total")
            eff = tue.get("tool_call_effective")
            llm_n = tue.get("llm_judged_count")
            rule_n = tue.get("rule_invalid_count")
            chunks.append(f"工具有效性=ratio={ratio} effective={eff}/{tot} rule_invalid={rule_n} llm_judged={llm_n}")
    else:
        chunks.append("工具有效性=?")

    er = base.get("evidence_recall")
    if not isinstance(er, dict):
        chunks.append("证据召回=-")
    elif er.get("skipped") or er.get("skip_reason") == "disabled":
        chunks.append("证据召回=关闭")
    else:
        rec = er.get("recall")
        hit = er.get("hit_count")
        gold = er.get("gold_total")
        chunks.append(f"证据召回=recall={rec} hit={hit}/{gold}")

    aim = base.get("answer_information_metrics")
    if aim is None or not isinstance(aim, dict):
        chunks.append("答案信息点=未启用")
    else:
        err = aim.get("error")
        sr = aim.get("skip_reason")
        if err:
            es = str(err).replace("\n", " ")
            chunks.append(f"答案信息点=失败({es[:100]}{'…' if len(es) > 100 else ''})")
        elif sr:
            chunks.append(f"答案信息点=跳过({sr})")
        else:
            chunks.append(
                "答案信息点="
                f"R={aim.get('recall')} P={aim.get('precision')} F1={aim.get('f1')} tp={aim.get('tp')}"
            )
    return " | ".join(chunks)


def _inject_evidence_recall(
    base: Dict[str, Any],
    *,
    task: TaskItem,
    turn: Optional[ConversationTurn],
    tasks_path: Path,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
) -> None:
    if not with_evidence_recall:
        n_gold = len(task.evidence_ids or [])
        base["evidence_recall"] = {
            "skipped": True,
            "skip_reason": "disabled",
            "gold_total": n_gold,
            "hit_count": 0,
            "recall": None,
            "hits": [],
            "misses": [],
            "per_evidence": [],
            "index_user_id": None,
            "fuzzy": evidence_fuzzy,
            "fuzzy_threshold": evidence_fuzzy_threshold,
        }
        return
    root = benchmark_data_root if benchmark_data_root is not None else resolve_benchmark_data_root()
    base["evidence_recall"] = compute_evidence_recall(
        turn,
        task,
        benchmark_data_root=root,
        tasks_path=tasks_path,
        fuzzy=evidence_fuzzy,
        fuzzy_threshold=evidence_fuzzy_threshold,
    )


def _inject_answer_information_metrics(
    base: Dict[str, Any],
    *,
    task: TaskItem,
    with_answer_information_metrics: bool,
    agent_answer: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    if not with_answer_information_metrics:
        return
    if dry_run:
        from eval.metrics.answer_information import answer_information_metrics_skipped

        base["answer_information_metrics"] = answer_information_metrics_skipped(
            skip_reason="dry_run",
            task_id=task.task_id,
        )
        return
    ag = base.get("agent_final_answer") if agent_answer is None else agent_answer
    if agent_answer is not None and not str(agent_answer).strip():
        ag = None
    base["answer_information_metrics"] = compute_answer_information_metrics(
        final_answer=task.expected_behavior or "",
        agent_answer=ag,
        task_id=task.task_id,
        user_query=task.prompt or "",
    )


def _eval_one_task(
    task: TaskItem,
    session,
    *,
    tasks_path: Path,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool = False,
    dry_run: bool,
    with_evidence_recall: bool = True,
    benchmark_data_root: Optional[Path] = None,
    evidence_fuzzy: bool = False,
    evidence_fuzzy_threshold: float = 0.86,
    with_answer_information_metrics: bool = False,
    task_index: Optional[int] = None,
    task_total: Optional[int] = None,
) -> Dict[str, Any]:
    m = match_task_to_session(task, session)
    logger.debug("评测匹配 task_id={} -> {}", task.task_id, m.status)
    base: Dict[str, Any] = {
        **task_fields_public(task),
        "match_status": m.status,
        "match_global_turn_index": m.global_turn_index,
        "match_business_turn_index": m.business_turn_index,
        "match_turn_index": m.business_turn_index,
        "matched_turn_source_log": m.turn.source_log_path if m.turn else None,
        "match_warnings": m.warnings,
        "errors": [],
        "agent_final_answer": m.turn.final_assistant_text() if m.turn else None,
        "agent_total_tokens": m.turn.agent_total_tokens() if m.turn else None,
        "agent_processing_seconds": m.turn.agent_processing_duration_seconds() if m.turn else None,
    }
    if m.status != "ok" or m.turn is None:
        base["skipped"] = True
        base["total_score_0_100"] = None
        base["passed"] = None
        base["judge"] = None
        base["tool_efficiency"] = None
        base["tool_use_effectiveness"] = None
        logger.warning(
            "任务跳过 task_id={} 原因=匹配失败 status={} warnings={}",
            task.task_id,
            m.status,
            m.warnings,
        )
        _inject_evidence_recall(
            base,
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            base,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        logger.debug(
            "评测扩展指标 task_id={} match_status={} skipped={} | {}",
            task.task_id,
            m.status,
            base.get("skipped"),
            _extended_metrics_log_fragment(base),
        )
        return base

    if dry_run:
        base["skipped"] = False
        base["total_score_0_100"] = None
        base["passed"] = None
        base["judge"] = None
        base["tool_efficiency"] = None
        base["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            base,
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            base,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
            dry_run=True,
        )
        logger.debug(
            "dry-run 扩展指标 task_id={} | {}",
            task.task_id,
            _extended_metrics_log_fragment(base),
        )
        return base

    if not config.llm_config_ready():
        logger.warning("任务跳过 task_id={} 原因=LLM 未配置", task.task_id)
        base["errors"].append(
            "LLM 未配置：请设置环境变量 LLM_API_KEY（及可选 LLM_BASE_URL、LLM_MODEL）"
        )
        base["skipped"] = True
        base["total_score_0_100"] = None
        base["passed"] = None
        base["judge"] = None
        base["tool_efficiency"] = None
        base["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            base,
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            base,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        logger.debug(
            "评测扩展指标 task_id={} match_status={} skipped={} | {}",
            task.task_id,
            m.status,
            base.get("skipped"),
            _extended_metrics_log_fragment(base),
        )
        return base

    try:
        ens = run_judge(task, m.turn, pass_threshold=task.pass_threshold)
    except Exception as e:
        logger.exception("Judge 失败 task_id={}", task.task_id)
        base["errors"].append(str(e))
        base["skipped"] = True
        base["total_score_0_100"] = None
        base["passed"] = None
        base["judge"] = None
        base["tool_efficiency"] = None
        base["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            base,
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            base,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        logger.debug(
            "评测扩展指标 task_id={} match_status={} skipped={} | {}",
            task.task_id,
            m.status,
            base.get("skipped"),
            _extended_metrics_log_fragment(base),
        )
        return base

    if not ens.ok:
        errs = ens.report.get("failure_errors") or ["所有 Judge 模型调用均失败"]
        if isinstance(errs, list):
            base["errors"].extend(errs)
        else:
            base["errors"].append(str(errs))
        base["skipped"] = True
        base["total_score_0_100"] = None
        base["passed"] = None
        base["judge"] = ens.report
        base["tool_efficiency"] = None
        base["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            base,
            task=task,
            turn=m.turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            base,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        logger.debug(
            "评测扩展指标 task_id={} match_status={} skipped={} | {}",
            task.task_id,
            m.status,
            base.get("skipped"),
            _extended_metrics_log_fragment(base),
        )
        return base

    mean_score = ens.mean_total_score_0_100
    agg_passed = ens.aggregate_passed
    q = mean_score / 100.0
    eff = compute_tool_efficiency(m.turn, q) if with_efficiency else None
    tue = None
    if with_tool_use_effectiveness:
        try:
            tue = compute_tool_use_effectiveness(m.turn, task)
        except Exception as e:
            logger.warning("tool_effectiveness 失败 task_id={} err={}", task.task_id, e)
            tue = {
                "error": str(e),
                "effective_tool_ratio": None,
                "tool_call_total": None,
                "tool_call_effective": None,
                "tool_call_evaluations": [],
            }

    base["skipped"] = False
    base["total_score_0_100"] = mean_score
    base["passed"] = agg_passed
    base["pass_threshold"] = task.pass_threshold
    base["judge"] = ens.report
    base["tool_efficiency"] = eff
    base["tool_use_effectiveness"] = tue
    _inject_evidence_recall(
        base,
        task=task,
        turn=m.turn,
        tasks_path=tasks_path,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
    )
    _inject_answer_information_metrics(
        base,
        task=task,
        with_answer_information_metrics=with_answer_information_metrics,
    )
    logger.info(
        "任务完成{} task_id={} mean_score={} aggregate_passed={}",
        f" [{task_index}/{task_total}]" if task_index is not None and task_total is not None else "",
        task.task_id,
        mean_score,
        agg_passed,
    )
    logger.debug(
        "任务扩展指标 task_id={} | {}",
        task.task_id,
        _extended_metrics_log_fragment(base),
    )
    return base


def _row_needs_metric_refresh(
    row: Dict[str, Any],
    *,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    with_evidence_recall: bool,
    with_answer_information_metrics: bool,
) -> bool:
    """当前 run 启用的可选指标在检查点行中是否缺失（需补算而非整行跳过）。"""
    if with_answer_information_metrics and "answer_information_metrics" not in row:
        return True
    if with_evidence_recall and "evidence_recall" not in row:
        return True

    scored_ok = (not row.get("skipped")) and row.get("total_score_0_100") is not None
    if with_tool_use_effectiveness and scored_ok:
        tu = row.get("tool_use_effectiveness", "__missing__")
        if tu is None or tu == "__missing__" or not isinstance(tu, dict):
            return True

    if with_efficiency and scored_ok:
        te = row.get("tool_efficiency", "__missing__")
        if te is None or te == "__missing__" or not isinstance(te, dict):
            return True

    return False


def _row_has_completed_judge(row: Dict[str, Any]) -> bool:
    """检查点行是否已有可复用的 Judge 结果（断点补算指标时不得清空）。"""
    if row.get("skipped"):
        return False
    if row.get("total_score_0_100") is None:
        return False
    if primary_judge_dict(row) is not None:
        return True
    jb = row.get("judge")
    return isinstance(jb, dict) and jb.get("mode") == "rule"


def _resolve_row_eval_log_spec(row: Dict[str, Any], task: TaskItem, default_spec: str) -> str:
    """补算指标时：优先用检查点行内 eval_log_spec（该题 Judge 所依日志）。"""
    for cand in (row.get("eval_log_spec"), task.session_log, default_spec):
        if cand and str(cand).strip():
            return str(cand).strip()
    return default_spec


def _log_spec_paths_equivalent(spec_a: str, spec_b: str, resolve_spec) -> bool:
    try:
        return _normalize_log_path_for_meta(resolve_spec(spec_a)) == _normalize_log_path_for_meta(
            resolve_spec(spec_b)
        )
    except Exception:
        return str(spec_a).strip() == str(spec_b).strip()


def _warn_row_eval_log_spec_vs_default(
    row: Dict[str, Any],
    task: TaskItem,
    default_spec: str,
    resolve_spec,
) -> None:
    """行内 eval_log_spec 与本次 --log 不一致时告警（不阻断续跑）。"""
    row_spec = str(row.get("eval_log_spec") or "").strip()
    if not row_spec:
        return
    if _log_spec_paths_equivalent(row_spec, default_spec, resolve_spec):
        return
    logger.warning(
        "batch 检查点 task_id={} 行内 eval_log_spec={!r} 与本次 --log={!r} 路径不一致；"
        "补算可选指标将使用行内日志。若需续跑整份检查点，请使 --log 与生成该 JSON 时一致，"
        "且各任务 eval_log_spec 通常应与顶层 log_path_spec 相同（除非任务单独指定 session_log）",
        task.task_id,
        row_spec,
        default_spec,
    )


def _patch_optional_metrics_on_row(
    out: Dict[str, Any],
    task: TaskItem,
    turn: Optional[ConversationTurn],
    *,
    tasks_path: Path,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    dry_run: bool,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
    with_answer_information_metrics: bool,
) -> None:
    """仅补算可选指标，不改动 judge / total_score / skipped。"""
    if dry_run:
        out["tool_efficiency"] = None
        out["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            out,
            task=task,
            turn=turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            out,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
            agent_answer=out.get("agent_final_answer"),
        )
        return

    if out.get("skipped") or out.get("total_score_0_100") is None:
        if "tool_efficiency" not in out:
            out["tool_efficiency"] = None
        if "tool_use_effectiveness" not in out:
            out["tool_use_effectiveness"] = None
        _inject_evidence_recall(
            out,
            task=task,
            turn=turn,
            tasks_path=tasks_path,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        )
        _inject_answer_information_metrics(
            out,
            task=task,
            with_answer_information_metrics=with_answer_information_metrics,
            agent_answer=out.get("agent_final_answer"),
        )
        return

    if with_efficiency and turn is not None:
        q = float(out["total_score_0_100"]) / 100.0
        out["tool_efficiency"] = compute_tool_efficiency(turn, q)
    if with_tool_use_effectiveness and turn is not None:
        try:
            out["tool_use_effectiveness"] = compute_tool_use_effectiveness(turn, task)
        except Exception as e:
            logger.warning("tool_effectiveness 检查点补算失败 task_id={} err={}", task.task_id, e)
            out["tool_use_effectiveness"] = {
                "error": str(e),
                "effective_tool_ratio": None,
                "tool_call_total": None,
                "tool_call_effective": None,
                "tool_call_evaluations": [],
            }

    _inject_evidence_recall(
        out,
        task=task,
        turn=turn,
        tasks_path=tasks_path,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
    )
    _inject_answer_information_metrics(
        out,
        task=task,
        with_answer_information_metrics=with_answer_information_metrics,
        agent_answer=out.get("agent_final_answer"),
    )


def _refresh_row_optional_metrics(
    row: Dict[str, Any],
    task: TaskItem,
    *,
    tasks_path: Path,
    default_spec: str,
    get_session,
    resolve_spec,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    dry_run: bool,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
    with_answer_information_metrics: bool,
) -> Dict[str, Any]:
    """
    在检查点行上补算新增的可选指标（不重跑 Judge）。
    已有 Judge 时：日志重匹配失败也保留原结果，仅更新可补算指标。
    无 Judge 时：匹配失败才回退整任务重评。
    """
    row = _merge_checkpoint_row_with_task(row, task)
    spec = _resolve_row_eval_log_spec(row, task, default_spec)
    _warn_row_eval_log_spec_vs_default(row, task, default_spec, resolve_spec)
    session = get_session(spec)

    if _row_has_completed_judge(row):
        out = copy.deepcopy(row)
        m = match_task_to_session(task, session)
        turn: Optional[ConversationTurn] = None
        if m.status == "ok" and m.turn is not None:
            turn = m.turn
            out["match_status"] = m.status
            out["match_global_turn_index"] = m.global_turn_index
            out["match_business_turn_index"] = m.business_turn_index
            out["match_turn_index"] = m.business_turn_index
            out["matched_turn_source_log"] = m.turn.source_log_path
            out["match_warnings"] = m.warnings
            out["agent_final_answer"] = m.turn.final_assistant_text()
            out["agent_total_tokens"] = m.turn.agent_total_tokens()
            out["agent_processing_seconds"] = m.turn.agent_processing_duration_seconds()
        else:
            logger.debug(
                "batch 检查点补算保留已有 Judge task_id={} log={!r} rematch={}",
                task.task_id,
                spec,
                m.status,
            )
        _patch_optional_metrics_on_row(
            out,
            task,
            turn,
            tasks_path=tasks_path,
            with_efficiency=with_efficiency,
            with_tool_use_effectiveness=with_tool_use_effectiveness,
            dry_run=dry_run,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        out["eval_log_spec"] = spec
        out["eval_log_target"] = str(resolve_spec(spec))
        out["eval_source_files"] = session.source_paths or []
        logger.info(
            "batch 检查点已补算新增指标(保留Judge) task_id={} | {}",
            task.task_id,
            _extended_metrics_log_fragment(out),
        )
        return out

    m = match_task_to_session(task, session)
    if m.status != "ok" or m.turn is None:
        logger.info("batch 检查点补算回退为全量重评 task_id={} reason=match_not_ok", task.task_id)
        out = _eval_one_task(
            task,
            session,
            tasks_path=tasks_path,
            with_efficiency=with_efficiency,
            with_tool_use_effectiveness=with_tool_use_effectiveness,
            dry_run=dry_run,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        out["eval_log_spec"] = spec
        out["eval_log_target"] = str(resolve_spec(spec))
        out["eval_source_files"] = session.source_paths or []
        return out

    if not row_eval_content_matches_task(row, task):
        logger.warning(
            "batch 检查点行与当前用例内容不一致，全量重评 task_id={}",
            task.task_id,
        )
        out = _eval_one_task(
            task,
            session,
            tasks_path=tasks_path,
            with_efficiency=with_efficiency,
            with_tool_use_effectiveness=with_tool_use_effectiveness,
            dry_run=dry_run,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
            with_answer_information_metrics=with_answer_information_metrics,
        )
        out["eval_log_spec"] = spec
        out["eval_log_target"] = str(resolve_spec(spec))
        out["eval_source_files"] = session.source_paths or []
        return out

    out = copy.deepcopy(row)
    out["match_status"] = m.status
    out["match_global_turn_index"] = m.global_turn_index
    out["match_business_turn_index"] = m.business_turn_index
    out["match_turn_index"] = m.business_turn_index
    out["matched_turn_source_log"] = m.turn.source_log_path if m.turn else None
    out["match_warnings"] = m.warnings
    out["agent_final_answer"] = m.turn.final_assistant_text() if m.turn else None
    out["agent_total_tokens"] = m.turn.agent_total_tokens() if m.turn else None
    out["agent_processing_seconds"] = m.turn.agent_processing_duration_seconds() if m.turn else None
    _patch_optional_metrics_on_row(
        out,
        task,
        m.turn,
        tasks_path=tasks_path,
        with_efficiency=with_efficiency,
        with_tool_use_effectiveness=with_tool_use_effectiveness,
        dry_run=dry_run,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        with_answer_information_metrics=with_answer_information_metrics,
    )
    out["eval_log_spec"] = spec
    out["eval_log_target"] = str(resolve_spec(spec))
    out["eval_source_files"] = session.source_paths or []
    logger.info(
        "batch 检查点已补算新增指标 task_id={} | {}",
        task.task_id,
        _extended_metrics_log_fragment(out),
    )
    return out


def _is_reusable_checkpoint_row(row: Dict[str, Any]) -> bool:
    if not row_eval_fingerprint(row):
        return False
    if row.get("judge") is not None:
        return True
    if row.get("skipped"):
        return True
    return row.get("total_score_0_100") is not None


def _checkpoint_resolve_task_rows(
    loaded_tasks: List[TaskItem],
    prev_tasks: Any,
) -> List[Optional[Dict[str, Any]]]:
    """
    按用例内容指纹从检查点匹配可复用行（与列表下标、task_id 无关）。
    返回与 loaded_tasks 等长的列表：可复用则为合并后的行，否则为 None。
    """
    if not isinstance(prev_tasks, list):
        return [None] * len(loaded_tasks)

    pool: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prev_tasks:
        if isinstance(row, dict) and _is_reusable_checkpoint_row(row):
            pool[row_eval_fingerprint(row)].append(row)

    resolved: List[Optional[Dict[str, Any]]] = []
    reused = 0
    for task in loaded_tasks:
        fp = task_eval_fingerprint(task)
        candidates = pool.get(fp) or []
        if not candidates:
            resolved.append(None)
            continue
        src = candidates.pop(0)
        merged = _merge_checkpoint_row_with_task(src, task)
        if src.get("task_id") != task.task_id:
            logger.info(
                "batch 检查点按内容指纹复用评测结果，task_id {} -> {}",
                src.get("task_id"),
                task.task_id,
            )
        resolved.append(merged)
        reused += 1

    if reused:
        logger.info(
            "batch 断点续跑: 按内容指纹复用 {} / {} 条（task_id 变更不影响匹配）",
            reused,
            len(loaded_tasks),
        )
    return resolved


def _checkpoint_apply_new_metrics_to_resolved(
    resolved: List[Optional[Dict[str, Any]]],
    loaded_tasks: List[TaskItem],
    *,
    default_spec: str,
    get_session,
    resolve_spec,
    tasks_path: Path,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    dry_run: bool,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
    with_answer_information_metrics: bool,
) -> List[Optional[Dict[str, Any]]]:
    if len(resolved) != len(loaded_tasks):
        return resolved
    updated: List[Optional[Dict[str, Any]]] = []
    for row, task in zip(resolved, loaded_tasks):
        if row is None:
            updated.append(None)
            continue
        if not _row_needs_metric_refresh(
            row,
            with_efficiency=with_efficiency,
            with_tool_use_effectiveness=with_tool_use_effectiveness,
            with_evidence_recall=with_evidence_recall,
            with_answer_information_metrics=with_answer_information_metrics,
        ):
            updated.append(copy.deepcopy(row))
            continue
        updated.append(
            _refresh_row_optional_metrics(
                row,
                task,
                tasks_path=tasks_path,
                default_spec=default_spec,
                get_session=get_session,
                resolve_spec=resolve_spec,
                with_efficiency=with_efficiency,
                with_tool_use_effectiveness=with_tool_use_effectiveness,
                dry_run=dry_run,
                with_evidence_recall=with_evidence_recall,
                benchmark_data_root=benchmark_data_root,
                evidence_fuzzy=evidence_fuzzy,
                evidence_fuzzy_threshold=evidence_fuzzy_threshold,
                with_answer_information_metrics=with_answer_information_metrics,
            )
        )
    return updated


def run_batch(
    tasks_path: Union[str, Path],
    *,
    log_override: Optional[str] = None,
    with_efficiency: bool = False,
    with_tool_use_effectiveness: bool = False,
    dry_run: bool = False,
    log_base_dir: Optional[Path] = None,
    with_evidence_recall: bool = True,
    benchmark_data_root: Optional[Path] = None,
    evidence_fuzzy: bool = False,
    evidence_fuzzy_threshold: float = 0.86,
    with_answer_information_metrics: bool = False,
    checkpoint_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    tasks_path = Path(tasks_path)
    loaded = load_tasks_document(tasks_path)
    logger.info(
        "已加载评测任务条数={} (任务文件格式={})",
        len(loaded.tasks),
        loaded.format,
    )
    er_root = benchmark_data_root if benchmark_data_root is not None else resolve_benchmark_data_root()
    preflight = _active_evidence_preflight(loaded.tasks, er_root)
    idx_uid = resolve_index_user_id(tasks_path)
    logger.info(
        "证据召回配置: benchmark_data_root={} index_user_id={} "
        "(覆盖: --benchmark-data-root / 环境 BENCHMARK_DATA_ROOT；用户: BENCHMARK_INDEX_USER_ID 或 tasks 路径推断)",
        str(er_root.resolve()) if er_root else None,
        idx_uid,
    )
    bases = _log_bases_for_tasks_file(tasks_path, loaded.format, log_base_dir)
    default_spec = _resolve_log_spec(
        loaded.log_path, log_override, tasks_path, loaded_format=loaded.format
    )
    get_session, resolve_spec = _session_getter_for_bases(bases)

    default_target = resolve_spec(default_spec)
    default_session = get_session(default_spec)
    # dry_run：不读断点、不做增量检查点写入；仍可在 persist_path 上于结束时一次性落盘。
    persist_path: Optional[Path] = Path(checkpoint_path).resolve() if checkpoint_path else None
    resume_path: Optional[Path] = persist_path if (persist_path is not None and not dry_run) else None
    if dry_run and persist_path:
        logger.info("batch dry_run：断点续跑与增量检查点关闭，全部任务结束后一次性写入 {}", persist_path)
    if resume_path and overwrite:
        logger.info("batch 检查点 overwrite=true，将从头测评并覆盖 {}", resume_path)
    elif resume_path and not overwrite:
        logger.info("batch 检查点路径={}（存在则按前缀续跑，除非参数不一致）", resume_path)

    logger.info(
        "batch 开始 path={} dry_run={} 默认日志={} | 默认会话: 全量交互轮次={} 业务向交互轮次={} | "
        "evidence_recall={} tool_use_effectiveness={} answer_information_metrics={} efficiency={}",
        tasks_path,
        dry_run,
        default_target,
        len(default_session.turns),
        len(default_session.business_turns()),
        with_evidence_recall,
        with_tool_use_effectiveness,
        with_answer_information_metrics,
        with_efficiency,
    )

    meta = _batch_checkpoint_meta(
        loaded=loaded,
        tasks_path=tasks_path,
        default_spec=default_spec,
        default_target=default_target,
        dry_run=dry_run,
        with_efficiency=with_efficiency,
        with_tool_use_effectiveness=with_tool_use_effectiveness,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        with_answer_information_metrics=with_answer_information_metrics,
    )

    resolved_rows: List[Optional[Dict[str, Any]]] = [None] * len(loaded.tasks)
    if resume_path and not overwrite:
        try:
            if resume_path.exists():
                prev = json.loads(resume_path.read_text(encoding="utf-8"))
                if isinstance(prev, dict) and _batch_checkpoint_compatible(prev, meta):
                    resolved_rows = _checkpoint_resolve_task_rows(
                        loaded.tasks, prev.get("tasks")
                    )
                elif isinstance(prev, dict):
                    logger.warning(
                        "batch 检查点与当前任务文件或测评参数不一致，将从头测评并覆盖: {}",
                        resume_path,
                    )
        except Exception as e:
            logger.warning("batch 检查点读取失败，将从头测评: {} err={}", resume_path, e)

    if any(r is not None for r in resolved_rows):
        resolved_rows = _checkpoint_apply_new_metrics_to_resolved(
            resolved_rows,
            loaded.tasks,
            default_spec=default_spec,
            get_session=get_session,
            resolve_spec=resolve_spec,
            tasks_path=tasks_path,
            with_efficiency=with_efficiency,
            with_tool_use_effectiveness=with_tool_use_effectiveness,
            dry_run=dry_run,
            with_evidence_recall=with_evidence_recall,
            benchmark_data_root=benchmark_data_root,
            evidence_fuzzy=evidence_fuzzy,
            evidence_fuzzy_threshold=evidence_fuzzy_threshold,
            with_answer_information_metrics=with_answer_information_metrics,
        )

    per_task: List[Dict[str, Any]] = []
    task_total = len(loaded.tasks)
    reused_total = sum(row is not None for row in resolved_rows)
    if show_progress:
        _print_batch_progress(
            f"[评测开始] 共 {task_total} 题 | 已复用 {reused_total} 题 | 待评 {task_total - reused_total} 题"
        )
    for idx, (task, reused) in enumerate(zip(loaded.tasks, resolved_rows), start=1):
        if reused is not None:
            row = reused
        else:
            if show_progress:
                _print_batch_progress(f"[{idx}/{task_total}] {task.task_id} 评测中...")
            spec = task.session_log or default_spec
            sess = get_session(spec)
            logger.debug(
                "batch 评测中 task_id={} session_log_spec={!r} -> {}",
                task.task_id,
                spec,
                resolve_spec(spec),
            )
            row = _eval_one_task(
                task,
                sess,
                tasks_path=tasks_path,
                with_efficiency=with_efficiency,
                with_tool_use_effectiveness=with_tool_use_effectiveness,
                dry_run=dry_run,
                with_evidence_recall=with_evidence_recall,
                benchmark_data_root=benchmark_data_root,
                evidence_fuzzy=evidence_fuzzy,
                evidence_fuzzy_threshold=evidence_fuzzy_threshold,
                with_answer_information_metrics=with_answer_information_metrics,
                task_index=idx,
                task_total=task_total,
            )
            row["eval_log_spec"] = spec
            row["eval_log_target"] = str(resolve_spec(spec))
            row["eval_source_files"] = sess.source_paths or []
            if show_progress:
                _print_batch_progress(
                    _format_batch_task_progress(idx, task_total, task.task_id, row)
                )
        per_task.append(row)
        if resume_path:
            out_partial = _assemble_batch_report(
                loaded=loaded,
                tasks_path=tasks_path,
                default_spec=default_spec,
                default_target=default_target,
                default_session=default_session,
                per_task=per_task,
                dry_run=dry_run,
                with_efficiency=with_efficiency,
                with_tool_use_effectiveness=with_tool_use_effectiveness,
                with_evidence_recall=with_evidence_recall,
                benchmark_data_root=benchmark_data_root,
                evidence_fuzzy=evidence_fuzzy,
                evidence_fuzzy_threshold=evidence_fuzzy_threshold,
                with_answer_information_metrics=with_answer_information_metrics,
            )
            atomic_write_json(resume_path, out_partial)
            logger.debug("batch 检查点已写入 {} 条任务 -> {}", len(per_task), resume_path)

    out = _assemble_batch_report(
        loaded=loaded,
        tasks_path=tasks_path,
        default_spec=default_spec,
        default_target=default_target,
        default_session=default_session,
        per_task=per_task,
        dry_run=dry_run,
        with_efficiency=with_efficiency,
        with_tool_use_effectiveness=with_tool_use_effectiveness,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        with_answer_information_metrics=with_answer_information_metrics,
    )
    if preflight is not None:
        out["evidence_preflight"] = preflight
    if persist_path:
        atomic_write_json(persist_path, out)

    sm = out["summary"]
    logger.info(
        "batch 结束 evaluated={} skipped={} mean_score={} pass_rate={}",
        sm.get("evaluated_count"),
        sm.get("skipped_count"),
        sm.get("mean_score"),
        sm.get("pass_rate"),
    )
    if show_progress:
        evaluated = int(sm.get("evaluated_count") or 0)
        passed = int(sm.get("passed_count") or 0)
        mean_score = sm.get("mean_score")
        mean_text = f"{float(mean_score):.2f}" if mean_score is not None else "-"
        _print_batch_progress(
            f"[评测完成] 有效 {evaluated}/{task_total} 题 | 均分 {mean_text} | 通过 {passed}/{evaluated}"
        )
    logger.info(
        "batch 扩展指标汇总 | 证据召回 mean_recall={} (evaluated={}, no_tool_call_excluded={}) | "
        "证据召回(含无工具调用) mean_recall_all={} (evaluated={}) | "
        "答案信息点 mean_F1={} mean_R={} mean_P={} (evaluated={}) | "
        "工具有效性 mean_effective_ratio={} (evaluated={}, no_tool_call_excluded={}) | "
        "工具有效性(含无工具调用) mean_effective_ratio_all={} (evaluated={})",
        sm.get("mean_evidence_recall"),
        sm.get("evidence_recall_evaluated_count"),
        sm.get("evidence_recall_no_tool_call_excluded_count"),
        sm.get("mean_evidence_recall_including_no_tool_call"),
        sm.get("evidence_recall_including_no_tool_call_evaluated_count"),
        sm.get("mean_answer_info_f1"),
        sm.get("mean_answer_info_recall"),
        sm.get("mean_answer_info_precision"),
        sm.get("answer_info_evaluated_count"),
        sm.get("mean_effective_tool_ratio"),
        sm.get("tool_use_effectiveness_evaluated_count"),
        sm.get("tool_use_effectiveness_no_tool_call_excluded_count"),
        sm.get("mean_effective_tool_ratio_including_no_tool_call"),
        sm.get("tool_use_effectiveness_including_no_tool_call_evaluated_count"),
    )
    return out


def run_one(
    tasks_path: Union[str, Path],
    *,
    task_id: Optional[str] = None,
    task_index: Optional[int] = None,
    log_override: Optional[str] = None,
    with_efficiency: bool = False,
    with_tool_use_effectiveness: bool = False,
    dry_run: bool = False,
    log_base_dir: Optional[Path] = None,
    with_evidence_recall: bool = True,
    benchmark_data_root: Optional[Path] = None,
    evidence_fuzzy: bool = False,
    evidence_fuzzy_threshold: float = 0.86,
    with_answer_information_metrics: bool = False,
) -> Dict[str, Any]:
    tasks_path = Path(tasks_path)
    loaded = load_tasks_document(tasks_path)
    logger.info(
        "已加载评测任务条数={} (任务文件格式={})",
        len(loaded.tasks),
        loaded.format,
    )
    er_root_1 = benchmark_data_root if benchmark_data_root is not None else resolve_benchmark_data_root()
    logger.info(
        "证据召回配置: benchmark_data_root={} index_user_id={}",
        str(er_root_1.resolve()) if er_root_1 else None,
        resolve_index_user_id(tasks_path),
    )
    if task_id:
        task = next((t for t in loaded.tasks if t.task_id == task_id), None)
        if task is None:
            raise ValueError(f"未找到 task_id={task_id!r}")
    elif task_index is not None:
        if task_index < 0 or task_index >= len(loaded.tasks):
            raise ValueError(f"task_index 越界: {task_index}")
        task = loaded.tasks[task_index]
    else:
        raise ValueError("必须指定 --task-id 或 --task-index")

    preflight = _active_evidence_preflight([task], er_root_1)

    bases = _log_bases_for_tasks_file(tasks_path, loaded.format, log_base_dir)
    get_session, resolve_spec = _session_getter_for_bases(bases)
    log_spec = log_override or task.session_log or loaded.log_path
    if not log_spec:
        raise ValueError("No agent trace was provided; use --log")
    logger.info(
        "run-one 指定 task_id={!r} task_index={!r} 实际评测 task_id={} 解析日志={}",
        task_id,
        task_index,
        task.task_id,
        resolve_spec(log_spec),
    )
    session = get_session(log_spec)
    logger.info(
        "run-one 当前会话: 全量交互轮次={} 业务向交互轮次={} session_id={}",
        len(session.turns),
        len(session.business_turns()),
        session.session_id,
    )
    row = _eval_one_task(
        task,
        session,
        tasks_path=tasks_path,
        with_efficiency=with_efficiency,
        with_tool_use_effectiveness=with_tool_use_effectiveness,
        dry_run=dry_run,
        with_evidence_recall=with_evidence_recall,
        benchmark_data_root=benchmark_data_root,
        evidence_fuzzy=evidence_fuzzy,
        evidence_fuzzy_threshold=evidence_fuzzy_threshold,
        with_answer_information_metrics=with_answer_information_metrics,
    )
    row["eval_log_spec"] = log_spec
    row["eval_log_target"] = str(resolve_spec(log_spec))
    row["eval_source_files"] = session.source_paths or []
    if preflight is not None:
        row["evidence_preflight"] = preflight
    return row


def _summarize(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [x for x in per_task if not x.get("skipped") and x.get("total_score_0_100") is not None]
    scores = [float(x["total_score_0_100"]) for x in valid]
    mean_score = sum(scores) / len(scores) if scores else None
    passed_n = sum(1 for x in valid if x.get("passed"))
    pass_rate = passed_n / len(valid) if valid else None

    by_cap: Dict[str, List[float]] = defaultdict(list)
    by_sub: Dict[str, List[float]] = defaultdict(list)
    by_dim: Dict[str, List[float]] = defaultdict(list)
    for x in valid:
        s = float(x["total_score_0_100"])
        by_cap[x["capability"]].append(s)
        by_sub[x["sub_capability"]].append(s)
        by_dim[x["dimension"]].append(s)

    def avg_map(m: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: round(sum(v) / len(v), 4) for k, v in m.items()}

    ai_recalls: List[float] = []
    ai_precisions: List[float] = []
    ai_f1s: List[float] = []
    for x in per_task:
        aim = x.get("answer_information_metrics")
        if not isinstance(aim, dict):
            continue
        if aim.get("error"):
            continue
        r = aim.get("recall")
        p = aim.get("precision")
        f = aim.get("f1")
        if isinstance(r, (int, float)) and isinstance(p, (int, float)) and isinstance(f, (int, float)):
            ai_recalls.append(float(r))
            ai_precisions.append(float(p))
            ai_f1s.append(float(f))

    answer_info_summary: Dict[str, Any] = {
        "answer_info_evaluated_count": len(ai_recalls),
        "mean_answer_info_recall": round(sum(ai_recalls) / len(ai_recalls), 4) if ai_recalls else None,
        "mean_answer_info_precision": round(sum(ai_precisions) / len(ai_precisions), 4)
        if ai_precisions
        else None,
        "mean_answer_info_f1": round(sum(ai_f1s) / len(ai_f1s), 4) if ai_f1s else None,
    }

    er_vals: List[float] = []
    er_vals_including_no_tool_call: List[float] = []
    er_no_tool_call_excluded_count = 0
    for x in per_task:
        er = x.get("evidence_recall")
        if not isinstance(er, dict):
            continue
        if er.get("skip_reason") == "disabled":
            continue
        r_all = er.get("recall_including_no_tool_call")
        if isinstance(r_all, (int, float)):
            er_vals_including_no_tool_call.append(float(r_all))
        r = er.get("recall")
        if isinstance(r, (int, float)):
            er_vals.append(float(r))
        else:
            dbg = er.get("debug") if isinstance(er.get("debug"), dict) else {}
            if dbg.get("has_tool_calls") is False:
                er_no_tool_call_excluded_count += 1
    evidence_recall_summary: Dict[str, Any] = {
        "evidence_recall_evaluated_count": len(er_vals),
        "mean_evidence_recall": round(sum(er_vals) / len(er_vals), 4) if er_vals else None,
        "evidence_recall_including_no_tool_call_evaluated_count": len(er_vals_including_no_tool_call),
        "mean_evidence_recall_including_no_tool_call": (
            round(sum(er_vals_including_no_tool_call) / len(er_vals_including_no_tool_call), 4)
            if er_vals_including_no_tool_call
            else None
        ),
        "evidence_recall_no_tool_call_excluded_count": er_no_tool_call_excluded_count,
    }

    tue_ratios: List[float] = []
    tue_ratios_including_no_tool_call: List[float] = []
    tue_no_tool_call_excluded_count = 0
    for x in per_task:
        tue = x.get("tool_use_effectiveness")
        if not isinstance(tue, dict) or tue.get("error"):
            continue
        r = tue.get("effective_tool_ratio")
        if not isinstance(r, (int, float)):
            continue
        rv = float(r)
        tue_ratios_including_no_tool_call.append(rv)
        t_total = tue.get("tool_call_total")
        if isinstance(t_total, (int, float)) and int(t_total) <= 0:
            tue_no_tool_call_excluded_count += 1
            continue
        tue_ratios.append(rv)
    tool_use_summary: Dict[str, Any] = {
        "tool_use_effectiveness_evaluated_count": len(tue_ratios),
        "mean_effective_tool_ratio": round(sum(tue_ratios) / len(tue_ratios), 4) if tue_ratios else None,
        "tool_use_effectiveness_including_no_tool_call_evaluated_count": len(
            tue_ratios_including_no_tool_call
        ),
        "mean_effective_tool_ratio_including_no_tool_call": (
            round(sum(tue_ratios_including_no_tool_call) / len(tue_ratios_including_no_tool_call), 4)
            if tue_ratios_including_no_tool_call
            else None
        ),
        "tool_use_effectiveness_no_tool_call_excluded_count": tue_no_tool_call_excluded_count,
    }

    from eval.core.capabilities import summarize_capability_item_stats

    capability_item_stats = summarize_capability_item_stats(per_task)

    return {
        "evaluated_count": len(valid),
        "skipped_count": sum(1 for x in per_task if x.get("skipped")),
        "mean_score": round(mean_score, 4) if mean_score is not None else None,
        "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
        "passed_count": passed_n,
        "by_capability": avg_map(by_cap),
        "by_sub_capability": avg_map(by_sub),
        "by_dimension": avg_map(by_dim),
        "capability_item_stats": capability_item_stats,
        **answer_info_summary,
        **evidence_recall_summary,
        **tool_use_summary,
    }


def write_report(data: Dict[str, Any], out_path: Union[str, Path]) -> None:
    atomic_write_json(Path(out_path), data)


def atomic_write_json(path: Path, data: Any) -> None:
    """原子写入 JSON（同盘 replace），避免写入中断导致结果文件损坏。"""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _batch_checkpoint_meta(
    *,
    loaded,
    tasks_path: Path,
    default_spec: str,
    default_target: Path,
    dry_run: bool,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
    with_answer_information_metrics: bool,
) -> Dict[str, Any]:
    br = benchmark_data_root
    bdr = str(br.resolve()) if br is not None else None
    if bdr is None:
        r = resolve_benchmark_data_root()
        bdr = str(r.resolve()) if r else None
    return {
        "schema_version": loaded.schema_version,
        "tasks_path": str(tasks_path.resolve()),
        "task_file_format": loaded.format,
        "dry_run": dry_run,
        "with_efficiency": with_efficiency,
        "with_tool_use_effectiveness": with_tool_use_effectiveness,
        "with_evidence_recall": with_evidence_recall,
        "benchmark_data_root": bdr,
        "evidence_root": _active_evidence_root_text(Path(bdr)) if bdr else None,
        "evidence_fuzzy": evidence_fuzzy,
        "evidence_fuzzy_threshold": evidence_fuzzy_threshold,
        "with_answer_information_metrics": with_answer_information_metrics,
        "log_path_spec": default_spec,
        "log_path": str(Path(default_target).resolve()),
    }


# 旧版报告 JSON 顶层可能缺失下列键；续跑时按当前 meta 的默认值对齐后再比较
_CHECKPOINT_META_DEFAULTS: Dict[str, Any] = {
    "evidence_fuzzy": False,
    "evidence_fuzzy_threshold": 0.86,
}


def _normalize_checkpoint_benchmark_data_root(value: Any) -> Optional[str]:
    """将 benchmark_data_root 规范为可比较的路径字符串（不做环境变量兜底）。"""
    if value is None or value == "":
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, ValueError):
        return str(value).strip() or None


def _normalize_log_path_for_meta(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, ValueError):
        return str(value).strip() or None


def _checkpoint_meta_value_equal(key: str, prev_v: Any, cur_v: Any) -> bool:
    if key == "benchmark_data_root":
        return _normalize_checkpoint_benchmark_data_root(prev_v) == _normalize_checkpoint_benchmark_data_root(
            cur_v
        )
    if key == "evidence_root":
        return _normalize_checkpoint_benchmark_data_root(prev_v) == _normalize_checkpoint_benchmark_data_root(
            cur_v
        )
    if key in ("log_path_spec", "log_path"):
        return _normalize_log_path_for_meta(prev_v) == _normalize_log_path_for_meta(cur_v)
    if prev_v is None and key in _CHECKPOINT_META_DEFAULTS:
        prev_v = _CHECKPOINT_META_DEFAULTS[key]
    return prev_v == cur_v


def _batch_checkpoint_compatible(prev: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    for k, v in meta.items():
        if k == "log_source_files":
            continue
        prev_v = prev.get(k)
        if not _checkpoint_meta_value_equal(k, prev_v, v):
            logger.debug("batch 检查点 meta 不一致 key={} prev={!r} cur={!r}", k, prev.get(k), v)
            return False
    return True


def _assemble_batch_report(
    *,
    loaded,
    tasks_path: Path,
    default_spec: str,
    default_target: Path,
    default_session: ParsedSession,
    per_task: List[Dict[str, Any]],
    dry_run: bool,
    with_efficiency: bool,
    with_tool_use_effectiveness: bool,
    with_evidence_recall: bool,
    benchmark_data_root: Optional[Path],
    evidence_fuzzy: bool,
    evidence_fuzzy_threshold: float,
    with_answer_information_metrics: bool,
) -> Dict[str, Any]:
    summary = _summarize(per_task)
    br = benchmark_data_root
    resolved_br = br if br is not None else resolve_benchmark_data_root()
    out: Dict[str, Any] = {
        "schema_version": loaded.schema_version,
        "tasks_path": str(tasks_path.resolve()),
        "task_file_format": loaded.format,
        "log_path_spec": default_spec,
        "log_path": str(Path(default_target).resolve()),
        "log_source_files": default_session.source_paths or [],
        "dry_run": dry_run,
        "with_efficiency": with_efficiency,
        "with_tool_use_effectiveness": with_tool_use_effectiveness,
        "with_evidence_recall": with_evidence_recall,
        "benchmark_data_root": str(resolved_br.resolve()) if resolved_br else None,
        "evidence_root": _active_evidence_root_text(resolved_br),
        "evidence_fuzzy": evidence_fuzzy,
        "evidence_fuzzy_threshold": evidence_fuzzy_threshold,
        "with_answer_information_metrics": with_answer_information_metrics,
        "tasks": per_task,
        "summary": summary,
        "task_count_expected": len(loaded.tasks),
        "batch_all_tasks_completed": len(per_task) == len(loaded.tasks),
    }
    if loaded.load_stats is not None:
        out["case_load_stats"] = loaded.load_stats
    if loaded.skipped_empty_prompt:
        out["skipped_empty_prompt"] = loaded.skipped_empty_prompt
    return out
