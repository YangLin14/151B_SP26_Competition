#!/usr/bin/env python3
"""Update the QLoRA Markdown results tracker from an eval result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TRACKER_PATH = Path("docs/QLORA_RESULTS_TRACKER.md")
EVAL_HEADER = (
    "| Eval id | Train run / model | Backend | Adapter? | Eval split | n | Prompt mode | "
    "Token budget | k | Temp / top_p / top_k | MCQ | Free-form | Overall | Avg tokens | "
    "Boxed | Truncated | Output path | Notes |"
)
EVAL_SEPARATOR = (
    "|---|---|---|---|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update docs/QLORA_RESULTS_TRACKER.md")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--tracker-path", default=str(DEFAULT_TRACKER_PATH))
    parser.add_argument("--eval-id", default=None)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_metadata(results_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    path = metadata_path or results_path.with_suffix(".metadata.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def md_cell(value: Any, *, code: bool = False) -> str:
    text = "TBD" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if code and text not in {"", "TBD", "pending"}:
        return f"`{text}`"
    return text


def accuracy_cell(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "0 / 0 (0.00%)"
    correct = sum(bool(row.get("correct")) for row in rows)
    total = len(rows)
    return f"{correct} / {total} ({correct / total * 100:.2f}%)"


def average_tokens(rows: list[dict[str, Any]]) -> str:
    tokens = [
        token
        for row in rows
        for token in row.get("tokens_per_sample", [])
        if token is not None
    ]
    if not tokens:
        return "TBD"
    return f"{sum(tokens) / len(tokens):.2f}"


def boxed_cell(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "0 / 0 (0.00%)"
    boxed = sum(any(item is not None for item in row.get("samples_boxed", [])) for row in rows)
    total = len(rows)
    return f"{boxed} / {total} ({boxed / total * 100:.2f}%)"


def truncated_cell(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "0 / 0 (0.00%)"
    any_truncated = 0
    all_truncated = 0
    for row in rows:
        finish_reasons = row.get("finish_reasons", [])
        truncated = [reason == "length" for reason in finish_reasons]
        any_truncated += int(any(truncated))
        all_truncated += int(bool(truncated) and all(truncated))
    total = len(rows)
    return (
        f"any {any_truncated} / {total} ({any_truncated / total * 100:.2f}%), "
        f"all {all_truncated} / {total} ({all_truncated / total * 100:.2f}%)"
    )


def infer_train_run(metadata: dict[str, Any]) -> str:
    adapter_path = metadata.get("adapter_path")
    if adapter_path:
        parts = Path(str(adapter_path)).parts
        if "outputs" in parts:
            idx = parts.index("outputs")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return str(adapter_path)
    return f"base {metadata.get('model_id', 'unknown')}"


def infer_token_budget(metadata: dict[str, Any]) -> str:
    if metadata.get("eval_backend") == "transformers" or "max_new_tokens" in metadata:
        return f"max input {metadata.get('max_input_length', 'TBD')}, max new {metadata.get('max_new_tokens', 'TBD')}"
    if "max_tokens" in metadata:
        return f"max tokens {metadata.get('max_tokens')}"
    return "TBD"


def infer_prompt_mode(metadata: dict[str, Any]) -> str:
    if metadata.get("enable_thinking") is True:
        return "thinking"
    if metadata.get("enable_thinking") is False:
        return "no thinking"
    return "unknown"


def make_eval_row(
    *,
    eval_id: str,
    results_path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    notes: str,
) -> str:
    mcq_rows = [row for row in rows if row.get("is_mcq")]
    free_rows = [row for row in rows if not row.get("is_mcq")]
    backend = metadata.get("eval_backend", "unknown")
    adapter = "yes" if metadata.get("adapter_path") else "no"
    sampling = (
        f"{metadata.get('temperature', 'TBD')} / "
        f"{metadata.get('top_p', 'TBD')} / "
        f"{metadata.get('top_k', 'TBD')}"
    )
    cells = [
        md_cell(eval_id, code=True),
        md_cell(infer_train_run(metadata), code=bool(metadata.get("adapter_path"))),
        md_cell(backend),
        md_cell(adapter),
        md_cell(metadata.get("data_path", "TBD"), code=True),
        md_cell(len(rows)),
        md_cell(infer_prompt_mode(metadata)),
        md_cell(infer_token_budget(metadata)),
        md_cell(metadata.get("k", "TBD")),
        md_cell(sampling),
        md_cell(accuracy_cell(mcq_rows)),
        md_cell(accuracy_cell(free_rows)),
        md_cell(accuracy_cell(rows)),
        md_cell(average_tokens(rows)),
        md_cell(boxed_cell(rows)),
        md_cell(truncated_cell(rows)),
        md_cell(str(results_path), code=True),
        md_cell(notes),
    ]
    return "| " + " | ".join(cells) + " |"


def find_eval_table(lines: list[str]) -> tuple[int, int]:
    header_idx = next((idx for idx, line in enumerate(lines) if line.strip() == EVAL_HEADER), -1)
    if header_idx == -1:
        raise ValueError("Could not find Eval Results table header in tracker.")

    end_idx = header_idx + 1
    while end_idx + 1 < len(lines) and lines[end_idx + 1].startswith("|"):
        end_idx += 1
    return header_idx, end_idx


def update_tracker(
    *,
    tracker_path: Path,
    results_path: Path,
    metadata_path: Path | None = None,
    eval_id: str | None = None,
    notes: str = "",
) -> None:
    rows = load_jsonl(results_path)
    metadata = load_metadata(results_path, metadata_path)
    if not eval_id:
        eval_id = results_path.stem

    new_row = make_eval_row(
        eval_id=eval_id,
        results_path=results_path,
        rows=rows,
        metadata=metadata,
        notes=notes,
    )

    text = tracker_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_idx, end_idx = find_eval_table(lines)
    data_start = header_idx + 2

    eval_id_cell = f"| `{eval_id}` "
    for idx in range(data_start, end_idx + 1):
        if lines[idx].startswith(eval_id_cell):
            lines[idx] = new_row
            break
    else:
        lines.insert(end_idx + 1, new_row)

    tracker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    update_tracker(
        tracker_path=Path(args.tracker_path),
        results_path=Path(args.results_path),
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        eval_id=args.eval_id,
        notes=args.notes,
    )
    print(f"Updated tracker: {args.tracker_path}")


if __name__ == "__main__":
    main()
