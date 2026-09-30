"""Command-line interface for MobileMem-Struct evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from eval.core.runner import run_batch, run_dry_run, run_one
from eval.utils.logging import setup_cli_logging


def _add_evidence_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--with-evidence-recall",
        dest="with_evidence_recall",
        action="store_true",
        help="compute evidence recall (default)",
    )
    group.add_argument(
        "--no-evidence-recall",
        dest="with_evidence_recall",
        action="store_false",
        help="skip evidence recall; preflight still runs when evidence is provided",
    )
    parser.set_defaults(with_evidence_recall=True)
    parser.add_argument(
        "--evidence-root",
        dest="benchmark_data_root",
        default=None,
        help="user directory containing the active app batch.json files",
    )


def _add_optional_metrics(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="compute tool efficiency",
    )
    parser.add_argument(
        "--with-tool-use-effectiveness",
        action="store_true",
        help="compute tool-call effectiveness with additional Judge calls",
    )
    parser.add_argument(
        "--with-answer-information-metrics",
        action="store_true",
        help="compute answer-information precision, recall, and F1",
    )


def _add_output_flags(parser: argparse.ArgumentParser, default_name: str) -> None:
    parser.add_argument("--out", help=f"JSON output path (default name: {default_name})")
    parser.add_argument("--out-dir", help=f"directory for {default_name}")


def _output_file(
    out: Optional[str], out_dir: Optional[str], default_name: str
) -> Optional[str]:
    if out and out_dir:
        raise ValueError("--out and --out-dir cannot be used together")
    if out:
        return out
    if out_dir:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / default_name)
    return None


def _evidence_kwargs(args: argparse.Namespace) -> dict:
    value = str(getattr(args, "benchmark_data_root", None) or "").strip()
    return {
        "with_evidence_recall": bool(args.with_evidence_recall),
        "benchmark_data_root": Path(value).expanduser().resolve() if value else None,
        "evidence_fuzzy": False,
        "evidence_fuzzy_threshold": 1.0,
    }


def _metric_kwargs(args: argparse.Namespace) -> dict:
    return {
        "with_efficiency": bool(getattr(args, "efficiency", False)),
        "with_tool_use_effectiveness": bool(
            getattr(args, "with_tool_use_effectiveness", False)
        ),
        "with_answer_information_metrics": bool(
            getattr(args, "with_answer_information_metrics", False)
        ),
    }


def _emit(data: dict, path: Optional[str]) -> int:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(destination)
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate mobile-memory agent traces on MobileMem-Struct"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tasks", required=True, help="case.csv")
    common.add_argument("--log", required=True, help="agent JSONL trace")
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v for INFO and -vv for DEBUG logs",
    )
    _add_evidence_flags(common)

    dry_run = subcommands.add_parser(
        "dry-run", parents=[common], help="validate cases, evidence, and trace matching"
    )
    _add_output_flags(dry_run, "dry_run.json")

    batch = subcommands.add_parser(
        "batch", parents=[common], help="evaluate every case with an LLM Judge"
    )
    _add_optional_metrics(batch)
    _add_output_flags(batch, "report.json")
    batch.add_argument(
        "--overwrite", action="store_true", help="ignore an existing checkpoint"
    )
    batch.add_argument(
        "--quiet-progress", action="store_true", help="hide per-case progress"
    )

    run_one_parser = subcommands.add_parser(
        "run-one", parents=[common], help="evaluate one case"
    )
    select = run_one_parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--task-id", help="case ID")
    select.add_argument("--task-index", type=int, help="zero-based case index")
    _add_optional_metrics(run_one_parser)
    _add_output_flags(run_one_parser, "task.json")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_cli_logging(args.verbose)
    try:
        if args.command == "dry-run":
            data = run_dry_run(
                args.tasks,
                args.log,
                **_evidence_kwargs(args),
            )
            return _emit(data, _output_file(args.out, args.out_dir, "dry_run.json"))

        if args.command == "batch":
            output = _output_file(args.out, args.out_dir, "report.json")
            data = run_batch(
                args.tasks,
                log_override=args.log,
                **_evidence_kwargs(args),
                **_metric_kwargs(args),
                checkpoint_path=output,
                overwrite=args.overwrite,
                show_progress=not args.quiet_progress,
            )
            return 0 if output else _emit(data, None)

        if args.command == "run-one":
            data = run_one(
                args.tasks,
                task_id=args.task_id,
                task_index=args.task_index,
                log_override=args.log,
                **_evidence_kwargs(args),
                **_metric_kwargs(args),
            )
            return _emit(data, _output_file(args.out, args.out_dir, "task.json"))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
