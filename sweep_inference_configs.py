#!/usr/bin/env python3
"""Run public validation sweeps for final inference settings."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from run_inference import run_inference


BASE_CONFIG: dict[str, Any] = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.72,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "retry_bad": True,
    "retry_k": 2,
    "retry_max_tokens": 4096,
    "generation_chunk_size": 32,
}

SWEEP_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "A_k3_tok4096",
        "description": "k=3, max_tokens=4096",
        "k": 3,
        "max_tokens": 4096,
    },
    {
        "name": "B_k5_tok4096",
        "description": "k=5, max_tokens=4096",
        "k": 5,
        "max_tokens": 4096,
    },
    {
        "name": "C_k7_tok4096_chunk16",
        "description": "k=7, max_tokens=4096, generation_chunk_size=16",
        "k": 7,
        "max_tokens": 4096,
        "generation_chunk_size": 16,
    },
    {
        "name": "D_k5_tok6144",
        "description": "k=5, max_tokens=6144",
        "k": 5,
        "max_tokens": 6144,
        "retry_max_tokens": 6144,
    },
    {
        "name": "E_k5_retry4",
        "description": "k=5, retry_k=4",
        "k": 5,
        "max_tokens": 4096,
        "retry_k": 4,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep public inference configs")
    parser.add_argument("--data-path", default="data/public.jsonl")
    parser.add_argument("--output-dir", default="results/sweeps/public_inference")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="Comma-separated config names to run.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    return parser.parse_args()


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.4f}"


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_result(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    score = metadata.get("score_summary") or {}
    overall = score.get("overall") or {}
    mcq = score.get("mcq") or {}
    free_form = score.get("free_form") or {}
    gen = metadata.get("generation_summary") or {}

    boxed_any = gen.get("boxed_any") or {}
    boxed_all = gen.get("boxed_all") or {}
    trunc_any = gen.get("truncated_any") or {}
    trunc_all = gen.get("truncated_all") or {}
    retried = gen.get("retried") or {}

    return {
        "name": config["name"],
        "description": config["description"],
        "overall_correct": overall.get("correct"),
        "overall_total": overall.get("total"),
        "overall_accuracy_pct": pct(overall.get("accuracy")),
        "mcq_correct": mcq.get("correct"),
        "mcq_total": mcq.get("total"),
        "mcq_accuracy_pct": pct(mcq.get("accuracy")),
        "free_correct": free_form.get("correct"),
        "free_total": free_form.get("total"),
        "free_accuracy_pct": pct(free_form.get("accuracy")),
        "boxed_any_pct": pct(boxed_any.get("rate")),
        "boxed_all_pct": pct(boxed_all.get("rate")),
        "truncated_any_pct": pct(trunc_any.get("rate")),
        "truncated_all_pct": pct(trunc_all.get("rate")),
        "retried_pct": pct(retried.get("rate")),
        "avg_samples_per_question": f"{gen.get('avg_samples_per_question', 0.0):.4f}",
        "avg_tokens_per_sample": f"{gen.get('avg_tokens_per_sample', 0.0):.4f}",
        "vote_status_counts": json.dumps(gen.get("vote_status_counts", {}), sort_keys=True),
        "sample_count_counts": json.dumps(gen.get("sample_count_counts", {}), sort_keys=True),
        "elapsed_minutes": f"{metadata.get('elapsed_seconds', 0.0) / 60:.4f}",
        "k": metadata.get("k"),
        "max_tokens": metadata.get("max_tokens"),
        "generation_chunk_size": metadata.get("generation_chunk_size"),
        "retry_k": metadata.get("retry_k"),
        "retry_max_tokens": metadata.get("retry_max_tokens"),
        "output_path": metadata.get("output_path"),
        "metadata_path": "",
    }


def score_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def num(key: str) -> float:
        value = row.get(key)
        if value in (None, ""):
            return -1.0
        return float(value)

    return (
        num("overall_accuracy_pct"),
        num("boxed_any_pct"),
        -num("truncated_any_pct"),
        -num("retried_pct"),
    )


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    best = max(rows, key=score_sort_key) if rows else None
    json_path.write_text(
        json.dumps({"best": best, "runs": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved sweep summary: {json_path}")
    print(f"Saved sweep CSV: {csv_path}")
    if best:
        print("\n=== Best Config ===")
        print(f"name: {best['name']}")
        print(f"description: {best['description']}")
        print(f"overall accuracy: {best['overall_accuracy_pct']}%")
        print(f"MCQ accuracy: {best['mcq_accuracy_pct']}%")
        print(f"Free-form accuracy: {best['free_accuracy_pct']}%")
        print(f"boxed_any: {best['boxed_any_pct']}%")
        print(f"truncated_any: {best['truncated_any_pct']}%")
        print("\nUse this private command shape with the winning parameters:")
        print(
            "python run_inference.py "
            "--data-path data/private.jsonl "
            "--output-path results/submission_final.csv "
            f"--k {best['k']} "
            f"--max-tokens {best['max_tokens']} "
            "--max-model-len 8192 "
            "--gpu-memory-utilization 0.72 "
            "--max-num-seqs 4 "
            "--max-num-batched-tokens 8192 "
            "--no-enable-prefix-caching "
            f"--generation-chunk-size {best['generation_chunk_size']} "
            "--retry-bad "
            f"--retry-k {best['retry_k']} "
            f"--retry-max-tokens {best['retry_max_tokens']}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}

    rows: list[dict[str, Any]] = []
    for config in SWEEP_CONFIGS:
        if selected and config["name"] not in selected:
            continue

        run_config = {**BASE_CONFIG, **config}
        name = run_config["name"]
        output_path = output_dir / f"{name}.csv"
        raw_path = output_dir / f"{name}.raw.jsonl"
        metadata_path = output_dir / f"{name}.metadata.json"

        print("\n" + "=" * 80)
        print(f"Running {name}: {run_config['description']}")
        print("=" * 80)

        if args.skip_existing and metadata_path.exists():
            print(f"Skipping existing run: {metadata_path}")
        else:
            started = time.time()
            run_inference(
                data_path=args.data_path,
                output_path=str(output_path),
                model_id=args.model_id,
                k=int(run_config["k"]),
                max_tokens=int(run_config["max_tokens"]),
                max_model_len=int(run_config["max_model_len"]),
                gpu_memory_utilization=float(run_config["gpu_memory_utilization"]),
                max_num_seqs=int(run_config["max_num_seqs"]),
                max_num_batched_tokens=int(run_config["max_num_batched_tokens"]),
                enable_prefix_caching=bool(run_config["enable_prefix_caching"]),
                enable_chunked_prefill=bool(run_config["enable_chunked_prefill"]),
                generation_chunk_size=int(run_config["generation_chunk_size"]),
                retry_bad=bool(run_config["retry_bad"]),
                retry_k=int(run_config["retry_k"]),
                retry_max_tokens=int(run_config["retry_max_tokens"]),
                raw_output_path=str(raw_path),
                metadata_path=str(metadata_path),
                limit=args.limit,
            )
            print(f"Finished {name} in {(time.time() - started) / 60:.2f} minutes")

        metadata = load_metadata(metadata_path)
        row = flatten_result(run_config, metadata)
        row["metadata_path"] = str(metadata_path)
        rows.append(row)
        write_summary(output_dir, rows)


if __name__ == "__main__":
    main()
