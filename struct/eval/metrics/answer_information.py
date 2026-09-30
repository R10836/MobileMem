"""基于 LLM 从参考答案与 Agent 最终回复抽取信息点并匹配，计算 recall / precision / F1。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from eval import config
from eval.judging.client import chat_completion_for_model, parse_json_loose

ANSWER_INFO_SYSTEM = """你是评测辅助工具，只做一件事：在**给定用户任务/查询**的前提下，从「参考答案」与「助手最终回复」中抽取**与该查询直接相关**、**可独立核验**的原子信息点，并判断金标信息点是否在助手侧被语义召回。

规则：
0. **相关性（必须）**：gold_points 与 agent_points 中的每一条都必须与用户任务/查询直接相关——有助于判断「在该查询下」助手是否答对、是否覆盖关键事实。**不要**抽取与当前用户查询无关的背景介绍、旁支话题、通用百科式句子；若参考答案或助手回复中含大量无关内容，请忽略它们。
1. gold_points：仅来自参考答案文本，且满足规则 0；每条是一句完整、可判断真假的陈述或数据点；互不重复；顺序与原文大致一致；不要编造原文没有的内容。
2. agent_points：仅来自助手回复，且满足规则 0；同样原子化；互不重复。
3. matches：对 gold_points 中的**每一个** gold 点给一条记录（gold_index 从 0 与 gold_points 对齐）。若某条 gold 在助手回复中有同义、改写或省略细节但核心事实一致的对应，则 recalled=true，并给出对应的 agent_index（agent_points 下标）；否则 recalled=false 且 agent_index 为 null。
4. 一对一：同一个 agent_index 最多只对应一条 recalled=true 的金标；优先保证更多 gold 被召回（若多条 gold 对应同一助手句，只选一条 recalled=true，其余 recalled=false）。
5. 仅输出一个 JSON 对象，不要 Markdown 代码块。字段名必须使用：gold_points, agent_points, matches。
6. matches 数组元素字段：gold_index（int）、agent_index（int 或 null）、recalled（bool）、brief_reason（string，可短句说明为何算召回或未召回）。

不要自行计算 recall/precision/f1，由程序根据你的结构化输出计算。
"""

_USER_QUERY_MAX = 6000


def _coerce_str_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        s = str(x).strip() if x is not None else ""
        if s:
            out.append(s)
    return out


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _coerce_index_alias(item: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        raw = item.get(key)
        if raw is None or str(raw).strip().lower() in ("null", "none", ""):
            continue
        value = _coerce_int(raw)
        if value is not None:
            return value
    return None


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    return False


def _normalize_matches(
    raw_matches: Any,
    gold_n: int,
    agent_n: int,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_matches, list):
        return []
    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(raw_matches):
        if not isinstance(m, dict):
            continue
        gi = _coerce_index_alias(m, "gold_index", "goldIndex")
        if gi is None:
            gi = i if gold_n and i < gold_n else None
        ai = _coerce_index_alias(m, "agent_index", "agentIndex")
        rec = _coerce_bool(m.get("recalled", m.get("recall")))
        reason = m.get("brief_reason") or m.get("reason") or ""
        rows.append(
            {
                "gold_index": gi,
                "agent_index": ai,
                "recalled": rec,
                "brief_reason": str(reason).strip() if reason else "",
            }
        )
    return rows


def compute_tp_one_to_one(
    gold_n: int,
    agent_n: int,
    matches: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    一对一贪心：按 gold_index 升序遍历 recalled 且 agent_index 合法的边；
    同一 gold 或同一 agent 已占用则跳过（保留先出现的记录）。
    """
    cand = []
    for m in matches:
        if not m.get("recalled"):
            continue
        gi = m.get("gold_index")
        ai = m.get("agent_index")
        if gi is None or ai is None:
            continue
        if not isinstance(gi, int) or not isinstance(ai, int):
            continue
        if gi < 0 or gi >= gold_n or ai < 0 or ai >= agent_n:
            continue
        cand.append((gi, ai, m))
    cand.sort(key=lambda x: (x[0], x[1]))
    used_g: set[int] = set()
    used_a: set[int] = set()
    paired: List[Dict[str, Any]] = []
    tp = 0
    for gi, ai, m in cand:
        if gi in used_g or ai in used_a:
            continue
        used_g.add(gi)
        used_a.add(ai)
        tp += 1
        paired.append({"gold_index": gi, "agent_index": ai, "brief_reason": m.get("brief_reason", "")})
    return tp, paired


def compute_metrics_from_llm_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对已解析的 LLM JSON 计算 TP / recall / precision / f1；用于单测与二次校验。
    """
    gold_points = _coerce_str_list(data.get("gold_points", data.get("goldPoints")))
    agent_points = _coerce_str_list(data.get("agent_points", data.get("agentPoints")))
    raw_m = data.get("matches", data.get("match", []))
    matches = _normalize_matches(raw_m, len(gold_points), len(agent_points))

    g_n, a_n = len(gold_points), len(agent_points)
    if g_n == 0:
        return {
            "gold_points": [],
            "agent_points": agent_points,
            "matches": matches,
            "gold_point_count": 0,
            "agent_point_count": a_n,
            "tp": None,
            "recall": None,
            "precision": None,
            "f1": None,
            "skip_reason": "no_gold_points",
            "matched_pairs": [],
        }

    if a_n == 0:
        return {
            "gold_points": gold_points,
            "agent_points": [],
            "matches": matches,
            "gold_point_count": g_n,
            "agent_point_count": 0,
            "tp": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "skip_reason": None,
            "matched_pairs": [],
        }

    tp, paired = compute_tp_one_to_one(g_n, a_n, matches)
    recall = tp / g_n
    precision = tp / a_n
    if recall + precision == 0:
        f1: Optional[float] = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "gold_points": gold_points,
        "agent_points": agent_points,
        "matches": matches,
        "gold_point_count": g_n,
        "agent_point_count": a_n,
        "tp": tp,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6) if f1 is not None else None,
        "skip_reason": None,
        "matched_pairs": paired,
    }


def _build_user_payload(
    final_answer: str,
    agent_answer: str,
    task_id: Optional[str],
    user_query: str,
) -> str:
    tid = f"\n【task_id】{task_id}\n" if task_id else ""
    uq = (user_query or "").strip()
    if len(uq) > _USER_QUERY_MAX:
        uq = uq[:_USER_QUERY_MAX] + "…"
    uq_block = uq if uq else "（未单独提供，请结合参考答案所体现的任务意图判断相关性。）"
    return f"""{tid}
【用户任务/查询】
{uq_block}

【参考答案】
{final_answer.strip()}

【助手最终回复】
{agent_answer.strip()}
"""


def run_answer_information_llm(
    final_answer: str,
    agent_answer: str,
    *,
    model: str,
    task_id: Optional[str] = None,
    user_query: str = "",
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": ANSWER_INFO_SYSTEM},
        {
            "role": "user",
            "content": _build_user_payload(final_answer, agent_answer, task_id, user_query),
        },
    ]
    logger.debug(
        "answer_recall LLM 请求 task_id={} model={} user_chars={} gold_chars={} agent_chars={}",
        task_id,
        model,
        len((user_query or "").strip()),
        len((final_answer or "").strip()),
        len((agent_answer or "").strip()),
    )
    raw = chat_completion_for_model(
        messages,
        model=model,
        temperature=0.15,
        response_format_json=True,
    )
    data = parse_json_loose(raw)
    if not isinstance(data, dict):
        raise ValueError(f"LLM 返回非 JSON 对象: {type(data).__name__}")
    out = compute_metrics_from_llm_payload(data)
    out["model"] = model
    out["llm_raw_keys"] = sorted(data.keys())
    logger.debug(
        "answer_recall LLM 返回 task_id={} model={} R={} P={} F1={} tp={} gold_pts={} agent_pts={}",
        task_id,
        model,
        out.get("recall"),
        out.get("precision"),
        out.get("f1"),
        out.get("tp"),
        out.get("gold_point_count"),
        out.get("agent_point_count"),
    )
    return out


def _answer_info_one_model(
    final_answer: str,
    agent_answer: str,
    *,
    model: str,
    task_id: Optional[str],
    user_query: str = "",
) -> tuple[str, Dict[str, Any] | None, str | None]:
    try:
        out = run_answer_information_llm(
            final_answer,
            agent_answer,
            model=model,
            task_id=task_id,
            user_query=user_query,
        )
        return model, out, None
    except Exception as e:
        logger.warning("answer_recall 模型失败 task_id={} model={} err={}", task_id, model, e)
        return model, None, str(e)


def _empty_shell(
    *,
    judge_models: List[str],
    skip_reason: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    mode = "multi" if len(judge_models) > 1 else "single"
    return {
        "mode": mode,
        "judge_models": judge_models,
        "gold_points": [],
        "agent_points": [],
        "matches": [],
        "matched_pairs": [],
        "gold_point_count": 0 if skip_reason == "no_gold_points" else None,
        "agent_point_count": 0 if skip_reason == "no_agent_answer" else None,
        "tp": None,
        "recall": None,
        "precision": None,
        "f1": None,
        "skip_reason": skip_reason,
        "aggregate": {
            "mean_recall": None,
            "mean_precision": None,
            "mean_f1": None,
            "mean_tp": None,
            "models_succeeded": 0,
            "models_total": len(judge_models),
        },
        "per_model": [],
        "failure_errors": [],
        "error": error,
    }


def answer_information_metrics_skipped(
    *,
    skip_reason: str,
    task_id: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """dry-run 或未配置 LLM 时占位，不调用 answer_information LLM。"""
    models = config.judge_model_names()
    out = _empty_shell(judge_models=models, skip_reason=skip_reason, error=error)
    if skip_reason == "dry_run":
        logger.debug("answer_recall 跳过 task_id={} reason=dry_run", task_id)
    return out


def compute_answer_information_metrics(
    *,
    final_answer: str,
    agent_answer: Optional[str],
    task_id: Optional[str] = None,
    user_query: str = "",
) -> Dict[str, Any]:
    """
    对单任务计算答案信息点指标；与 Judge 共用 ``config.judge_model_names()``，
    多模型时并行调用并对 recall/precision/f1/tp 取算术平均。
    final_answer：TaskItem.expected_behavior 中的标准答案。
    user_query：用户任务/查询（通常为 TaskItem.prompt），用于约束仅抽取与查询相关的原子点。
    """
    models = config.judge_model_names()
    fa = (final_answer or "").strip()
    ag = (agent_answer or "").strip()

    if not fa:
        logger.debug("answer_recall 跳过 task_id={} reason=no_gold_points", task_id)
        base = _empty_shell(judge_models=models, skip_reason="no_gold_points", error=None)
        base["gold_point_count"] = 0
        base["agent_point_count"] = 0
        base["tp"] = None
        return base

    if not ag:
        logger.debug("answer_recall 跳过 task_id={} reason=no_agent_answer", task_id)
        base = _empty_shell(judge_models=models, skip_reason="no_agent_answer", error=None)
        base["gold_point_count"] = None
        base["agent_point_count"] = 0
        base["tp"] = 0
        base["recall"] = 0.0
        base["precision"] = 0.0
        base["f1"] = 0.0
        base["aggregate"] = {
            "mean_recall": 0.0,
            "mean_precision": 0.0,
            "mean_f1": 0.0,
            "mean_tp": 0.0,
            "models_succeeded": 0,
            "models_total": len(models),
        }
        return base

    if not config.llm_config_ready():
        logger.warning(
            "answer_recall 跳过 task_id={} reason=llm_not_configured",
            task_id,
        )
        return _empty_shell(
            judge_models=models,
            skip_reason=None,
            error="LLM 未配置：无法计算答案信息点指标（请设置 LLM_API_KEY、LLM_BASE_URL）",
        )

    per_model_rows: List[Dict[str, Any]] = []
    successes: List[Dict[str, Any]] = []
    failure_msgs: List[str] = []
    max_workers = min(len(models), config.judge_parallel_max_workers())
    logger.debug(
        "answer_recall 执行开始 task_id={} models={} parallel_workers={} gold_chars={} agent_chars={}",
        task_id,
        ",".join(models),
        max_workers if len(models) > 1 else 1,
        len(fa),
        len(ag),
    )

    def run_m(m: str) -> tuple[str, Dict[str, Any] | None, str | None]:
        return _answer_info_one_model(fa, ag, model=m, task_id=task_id, user_query=user_query)

    if len(models) == 1:
        model, out, err = run_m(models[0])
        if out is not None:
            row = {
                "model": model,
                "ok": True,
                "recall": out.get("recall"),
                "precision": out.get("precision"),
                "f1": out.get("f1"),
                "tp": out.get("tp"),
                "gold_point_count": out.get("gold_point_count"),
                "agent_point_count": out.get("agent_point_count"),
                "gold_points": out.get("gold_points"),
                "agent_points": out.get("agent_points"),
                "matches": out.get("matches"),
                "matched_pairs": out.get("matched_pairs"),
                "skip_reason": out.get("skip_reason"),
                "llm_raw_keys": out.get("llm_raw_keys"),
                "error": None,
            }
            per_model_rows.append(row)
            successes.append(out)
        else:
            per_model_rows.append(
                {
                    "model": model,
                    "ok": False,
                    "recall": None,
                    "precision": None,
                    "f1": None,
                    "tp": None,
                    "gold_point_count": None,
                    "agent_point_count": None,
                    "gold_points": [],
                    "agent_points": [],
                    "matches": [],
                    "matched_pairs": [],
                    "skip_reason": None,
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
                            "recall": out.get("recall"),
                            "precision": out.get("precision"),
                            "f1": out.get("f1"),
                            "tp": out.get("tp"),
                            "gold_point_count": out.get("gold_point_count"),
                            "agent_point_count": out.get("agent_point_count"),
                            "gold_points": out.get("gold_points"),
                            "agent_points": out.get("agent_points"),
                            "matches": out.get("matches"),
                            "matched_pairs": out.get("matched_pairs"),
                            "skip_reason": out.get("skip_reason"),
                            "llm_raw_keys": out.get("llm_raw_keys"),
                            "error": None,
                        }
                    )
                    successes.append(out)
                else:
                    per_model_rows.append(
                        {
                            "model": model,
                            "ok": False,
                            "recall": None,
                            "precision": None,
                            "f1": None,
                            "tp": None,
                            "gold_point_count": None,
                            "agent_point_count": None,
                            "gold_points": [],
                            "agent_points": [],
                            "matches": [],
                            "matched_pairs": [],
                            "skip_reason": None,
                            "error": err,
                        }
                    )
                    if err:
                        failure_msgs.append(f"{model}: {err}")
        per_model_rows.sort(key=lambda r: models.index(r["model"]) if r["model"] in models else 999)

    mode = "multi" if len(models) > 1 else "single"

    if not successes:
        logger.warning(
            "answer_recall 全部模型失败 task_id={} failures={}",
            task_id,
            failure_msgs,
        )
        return {
            "mode": mode,
            "judge_models": models,
            "gold_points": [],
            "agent_points": [],
            "matches": [],
            "matched_pairs": [],
            "gold_point_count": None,
            "agent_point_count": None,
            "tp": None,
            "recall": None,
            "precision": None,
            "f1": None,
            "skip_reason": None,
            "aggregate": {
                "mean_recall": None,
                "mean_precision": None,
                "mean_f1": None,
                "mean_tp": None,
                "models_succeeded": 0,
                "models_total": len(models),
            },
            "per_model": per_model_rows,
            "failure_errors": failure_msgs,
            "error": "; ".join(failure_msgs) if failure_msgs else "所有模型调用均失败",
        }

    recalls = [float(s["recall"]) for s in successes if s.get("recall") is not None]
    precs = [float(s["precision"]) for s in successes if s.get("precision") is not None]
    f1s = [float(s["f1"]) for s in successes if s.get("f1") is not None]
    tps = [float(s["tp"]) for s in successes if s.get("tp") is not None]

    mean_r = sum(recalls) / len(recalls) if recalls else None
    mean_p = sum(precs) / len(precs) if precs else None
    mean_f = sum(f1s) / len(f1s) if f1s else None
    mean_tp = sum(tps) / len(tps) if tps else None

    primary_row: Optional[Dict[str, Any]] = None
    for mid in models:
        for r in per_model_rows:
            if r.get("model") == mid and r.get("ok"):
                primary_row = r
                break
        if primary_row is not None:
            break
    if primary_row is None:
        primary_row = next((r for r in per_model_rows if r.get("ok")), {})
    gold_pts = list(primary_row.get("gold_points") or [])
    agent_pts = list(primary_row.get("agent_points") or [])
    matches_top = list(primary_row.get("matches") or [])
    pairs_top = list(primary_row.get("matched_pairs") or [])

    logger.debug(
        "answer_recall 聚合完成 task_id={} models_ok={}/{} mean_R={} mean_P={} mean_F1={} mean_tp={} "
        "primary_model={}",
        task_id,
        len(successes),
        len(models),
        round(mean_r, 6) if mean_r is not None else None,
        round(mean_p, 6) if mean_p is not None else None,
        round(mean_f, 6) if mean_f is not None else None,
        round(mean_tp, 4) if mean_tp is not None else None,
        primary_row.get("model"),
    )
    return {
        "mode": mode,
        "judge_models": models,
        "gold_points": gold_pts,
        "agent_points": agent_pts,
        "matches": matches_top,
        "matched_pairs": pairs_top,
        "gold_point_count": primary_row.get("gold_point_count"),
        "agent_point_count": primary_row.get("agent_point_count"),
        "tp": round(mean_tp, 4) if mean_tp is not None else None,
        "recall": round(mean_r, 6) if mean_r is not None else None,
        "precision": round(mean_p, 6) if mean_p is not None else None,
        "f1": round(mean_f, 6) if mean_f is not None else None,
        "skip_reason": None,
        "aggregate": {
            "mean_recall": round(mean_r, 6) if mean_r is not None else None,
            "mean_precision": round(mean_p, 6) if mean_p is not None else None,
            "mean_f1": round(mean_f, 6) if mean_f is not None else None,
            "mean_tp": round(mean_tp, 4) if mean_tp is not None else None,
            "models_succeeded": len(successes),
            "models_total": len(models),
        },
        "per_model": per_model_rows,
        "failure_errors": failure_msgs,
        "error": None,
    }
