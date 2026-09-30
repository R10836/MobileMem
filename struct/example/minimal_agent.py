"""A deliberately small, vendor-neutral agent for producing benchmark traces.

The example reads only each case's Query and the active app records. It never
loads GT, gold answers, scoring rules, or benchmark evidence IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import httpx

from eval import config  # Loads struct/.env when present.
from eval.evidence.catalog import load_active_evidence_index


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    ascii_words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    cjk_terms = {cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))}
    return ascii_words | cjk_terms


def _record_text(entry: dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)


def retrieve(
    query: str, evidence_index: dict[str, Any], *, top_k: int
) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for evidence_id, entry in evidence_index.get("by_evidence_id", {}).items():
        if not isinstance(entry, dict):
            continue
        score = len(query_tokens & _tokens(_record_text(entry)))
        if score > 0:
            ranked.append((score, str(evidence_id), entry))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "evidence_id": evidence_id,
            "data_type": entry.get("data_type"),
            "record": entry.get("record"),
        }
        for _, evidence_id, entry in ranked[:top_k]
    ]


def _read_queries(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            task_id = str(row.get("Query编号") or "").strip()
            query = str(row.get("Query") or "").strip()
            if not task_id or not query:
                raise ValueError(f"Missing Query编号 or Query at {path}:{row_number}")
            yield task_id, query


def _agent_settings() -> tuple[str, str, str]:
    api_key = os.environ.get("AGENT_API_KEY") or config.LLM_API_KEY
    base_url = os.environ.get("AGENT_BASE_URL") or config.LLM_BASE_URL
    model = os.environ.get("AGENT_MODEL") or config.LLM_MODEL
    if not api_key:
        raise ValueError("Set AGENT_API_KEY in struct/.env")
    return api_key, base_url.rstrip("/"), model


def answer(query: str, records: list[dict[str, Any]]) -> str:
    api_key, base_url, model = _agent_settings()
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the supplied device records. "
                    "Be concise. If the records are insufficient, say so clearly. "
                    "Do not claim that an action was completed."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"query": query, "records": records}, ensure_ascii=False
                ),
            },
        ],
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"] or "").strip()


def _write_jsonl_line(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run(cases: Path, evidence_root: Path, output: Path, *, top_k: int) -> None:
    evidence_index = load_active_evidence_index(evidence_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for task_id, query in _read_queries(cases):
            records = retrieve(query, evidence_index, top_k=top_k)
            call_id = f"search_{task_id}"
            _write_jsonl_line(handle, {"role": "user", "content": query})
            _write_jsonl_line(
                handle,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "search_memory",
                                "arguments": json.dumps(
                                    {"query": query, "top_k": top_k},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                },
            )
            _write_jsonl_line(
                handle,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "search_memory",
                    "content": json.dumps({"results": records}, ensure_ascii=False),
                },
            )
            _write_jsonl_line(handle, {"role": "assistant", "content": answer(query, records)})
            print(f"[{task_id}] wrote {len(records)} retrieved records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal Struct example agent")
    parser.add_argument("--cases", type=Path, required=True, help="case.csv")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output JSONL trace")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(
        args.cases.expanduser().resolve(),
        args.evidence_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        top_k=max(1, args.top_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
