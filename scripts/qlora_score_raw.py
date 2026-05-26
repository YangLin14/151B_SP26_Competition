#!/usr/bin/env python3
"""Score saved QLoRA raw generations without running model inference again."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved raw QLoRA generations")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tracker-path", default="docs/QLORA_RESULTS_TRACKER.md")
    parser.add_argument("--tracker-eval-id", default=None)
    parser.add_argument("--tracker-notes", default="")
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the Markdown QLoRA results tracker after scoring.",
    )
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


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(script_dir))

    from qlora_update_tracker import update_tracker
    from qlora_vllm_eval import print_summary, score_results

    eval_data = load_jsonl(Path(args.data_path))
    raw_rows = load_jsonl(Path(args.raw_path))

    eval_data = eval_data[: len(raw_rows)]
    per_question_raw = [row["samples"] for row in raw_rows]

    results = score_results(eval_data, per_question_raw)
    print_summary(results)
    write_jsonl(Path(args.output_path), results)

    metadata = {
        "data_path": args.data_path,
        "raw_path": args.raw_path,
        "output_path": args.output_path,
        "n_eval": len(results),
    }
    Path(args.output_path).with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved scored results to {args.output_path}")

    if args.update_tracker:
        try:
            update_tracker(
                tracker_path=Path(args.tracker_path),
                results_path=Path(args.output_path),
                metadata_path=Path(args.output_path).with_suffix(".metadata.json"),
                eval_id=args.tracker_eval_id,
                notes=args.tracker_notes,
            )
            print(f"Updated tracker: {args.tracker_path}")
        except Exception as exc:
            print(f"Warning: failed to update tracker: {exc}")


if __name__ == "__main__":
    main()
