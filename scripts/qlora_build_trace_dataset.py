#!/usr/bin/env python3
"""Build a QLoRA SFT trace dataset from scored public generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered public correct traces")
    parser.add_argument("--data-path", required=True, help="Original public split JSONL used for eval.")
    parser.add_argument("--results-path", required=True, help="Scored eval results JSONL.")
    parser.add_argument("--output-path", required=True, help="Filtered trace dataset JSONL.")
    parser.add_argument("--max-traces", type=int, default=-1)
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--require-boxed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-truncated", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_truncated(row: dict[str, Any]) -> bool:
    finish_reasons = row.get("finish_reasons", [])
    return any(reason == "length" for reason in finish_reasons)


def main() -> None:
    args = parse_args()

    source_rows = load_jsonl(Path(args.data_path))
    result_rows = load_jsonl(Path(args.results_path))
    source_by_id = {row.get("id"): row for row in source_rows}

    kept = []
    skipped = {
        "incorrect": 0,
        "missing_source": 0,
        "missing_boxed": 0,
        "short_response": 0,
        "truncated": 0,
    }

    for result in result_rows:
        if not bool(result.get("correct")):
            skipped["incorrect"] += 1
            continue
        source = source_by_id.get(result.get("id"))
        if source is None:
            skipped["missing_source"] += 1
            continue
        if args.require_boxed and result.get("voted") is None:
            skipped["missing_boxed"] += 1
            continue
        if not args.allow_truncated and is_truncated(result):
            skipped["truncated"] += 1
            continue

        response = str(result.get("rep_response") or "").strip()
        if len(response) < args.min_response_chars:
            skipped["short_response"] += 1
            continue

        kept.append(
            {
                "id": source.get("id"),
                "question": source.get("question"),
                "options": source.get("options"),
                "answer": source.get("answer"),
                "response": response,
                "voted": result.get("voted"),
                "is_mcq": bool(source.get("options")),
                "source_data_path": args.data_path,
                "source_results_path": args.results_path,
            }
        )

        if args.max_traces > 0 and len(kept) >= args.max_traces:
            break

    write_jsonl(Path(args.output_path), kept)

    print(f"Loaded source rows: {len(source_rows)}")
    print(f"Loaded result rows: {len(result_rows)}")
    print(f"Kept traces: {len(kept)}")
    for key, value in skipped.items():
        print(f"Skipped {key}: {value}")
    print(f"Saved traces to {args.output_path}")


if __name__ == "__main__":
    main()
