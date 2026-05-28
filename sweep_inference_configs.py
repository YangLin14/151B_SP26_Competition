#!/usr/bin/env python3
"""Run public validation sweeps for final inference settings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_CONFIG: dict[str, Any] = {
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 16384,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "retry_bad": True,
    "retry_k": 2,
    "retry_max_tokens": 32768,
    "generation_chunk_size": 32,
}

SWEEP_PRESETS: dict[str, list[dict[str, Any]]] = {
    "a30-long": [
        {
            "name": "A_k5_tok24576",
            "description": "current strong baseline: k=5, max_tokens=24576",
            "k": 5,
            "max_tokens": 24576,
        },
        {
            "name": "B_k7_tok24576_chunk16_seq6",
            "description": "more self-consistency: k=7, max_tokens=24576, chunk=16, seqs=6",
            "k": 7,
            "max_tokens": 24576,
            "max_num_seqs": 6,
            "generation_chunk_size": 16,
        },
        {
            "name": "C_k3_tok24576",
            "description": "cheaper long-thinking baseline: k=3, max_tokens=24576",
            "k": 3,
            "max_tokens": 24576,
        },
        {
            "name": "D_k5_tok16384",
            "description": "middle token budget: k=5, max_tokens=16384",
            "k": 5,
            "max_tokens": 16384,
            "retry_max_tokens": 24576,
        },
        {
            "name": "E_k5_tok24576_retry4",
            "description": "stronger adaptive retry: k=5, max_tokens=24576, retry_k=4",
            "k": 5,
            "max_tokens": 24576,
            "retry_k": 4,
        },
    ],
    "quick-short": [
        {
            "name": "A_k3_tok4096",
            "description": "fast short baseline: k=3, max_tokens=4096",
            "k": 3,
            "max_tokens": 4096,
            "retry_max_tokens": 8192,
        },
        {
            "name": "B_k5_tok4096",
            "description": "fast self-consistency: k=5, max_tokens=4096",
            "k": 5,
            "max_tokens": 4096,
            "retry_max_tokens": 8192,
        },
        {
            "name": "C_k7_tok4096_chunk16_seq6",
            "description": "fast k=7: max_tokens=4096, chunk=16, seqs=6",
            "k": 7,
            "max_tokens": 4096,
            "max_num_seqs": 6,
            "generation_chunk_size": 16,
            "retry_max_tokens": 8192,
        },
        {
            "name": "D_k5_tok6144",
            "description": "slightly longer short baseline: k=5, max_tokens=6144",
            "k": 5,
            "max_tokens": 6144,
            "retry_max_tokens": 12288,
        },
        {
            "name": "E_k5_tok4096_retry4",
            "description": "fast stronger adaptive retry: k=5, max_tokens=4096, retry_k=4",
            "k": 5,
            "max_tokens": 4096,
            "retry_k": 4,
            "retry_max_tokens": 8192,
        },
    ],
}

SWEEP_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "A_k5_tok24576",
        "description": "current strong baseline: k=5, max_tokens=24576",
        "k": 5,
        "max_tokens": 24576,
    },
    {
        "name": "B_k7_tok24576_chunk16_seq6",
        "description": "more self-consistency: k=7, max_tokens=24576, chunk=16, seqs=6",
        "k": 7,
        "max_tokens": 24576,
        "max_num_seqs": 6,
        "generation_chunk_size": 16,
    },
    {
        "name": "C_k3_tok24576",
        "description": "cheaper long-thinking baseline: k=3, max_tokens=24576",
        "k": 3,
        "max_tokens": 24576,
    },
    {
        "name": "D_k5_tok16384",
        "description": "middle token budget: k=5, max_tokens=16384",
        "k": 5,
        "max_tokens": 16384,
        "retry_max_tokens": 24576,
    },
    {
        "name": "E_k5_tok24576_retry4",
        "description": "stronger adaptive retry: k=5, max_tokens=24576, retry_k=4",
        "k": 5,
        "max_tokens": 24576,
        "retry_k": 4,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep public inference configs")
    parser.add_argument("--data-path", default="data/public.jsonl")
    parser.add_argument("--output-dir", default="results/sweeps/public_inference")
    parser.add_argument(
        "--preset",
        choices=sorted(SWEEP_PRESETS),
        default="a30-long",
        help="Config preset to run. a30-long is the recommended native-vLLM A30 sweep.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--only", default=None, help="Comma-separated config names to run.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=10.0,
        help="Pause between candidates so CUDA/NCCL cleanup settles before the next vLLM process.",
    )
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
    think_end_any = gen.get("think_end_any") or {}
    think_end_all = gen.get("think_end_all") or {}
    thinking_tokens = gen.get("thinking_tokens") or {}
    trunc_any = gen.get("truncated_any") or {}
    trunc_all = gen.get("truncated_all") or {}
    retried = gen.get("retried") or {}

    return {
        "name": config["name"],
        "description": config["description"],
        "status": metadata.get("status", "ok"),
        "error": metadata.get("error", ""),
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
        "think_end_any_pct": pct(think_end_any.get("rate")),
        "think_end_all_pct": pct(think_end_all.get("rate")),
        "thinking_tokens_avg": f"{thinking_tokens.get('avg', 0.0):.4f}",
        "thinking_tokens_p50": f"{thinking_tokens.get('p50', 0.0):.4f}",
        "thinking_tokens_p90": f"{thinking_tokens.get('p90', 0.0):.4f}",
        "thinking_tokens_p95": f"{thinking_tokens.get('p95', 0.0):.4f}",
        "thinking_tokens_max": thinking_tokens.get("max", 0),
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
        "max_num_seqs": metadata.get("max_num_seqs"),
        "max_num_batched_tokens": metadata.get("max_num_batched_tokens"),
        "start_index": metadata.get("start_index"),
        "end_index": metadata.get("end_index"),
        "limit": metadata.get("limit"),
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

    successful_rows = [row for row in rows if row.get("status") == "ok"]
    best = max(successful_rows, key=score_sort_key) if successful_rows else None
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
        print(f"think_end_any: {best['think_end_any_pct']}%")
        print(f"thinking_tokens_p95: {best['thinking_tokens_p95']}")
        print(f"truncated_any: {best['truncated_any_pct']}%")
        print("\nUse this private command shape with the winning parameters:")
        print(
            "python run_inference.py "
            "--data-path data/private.jsonl "
            "--output-path results/submission_final.csv "
            f"--k {best['k']} "
            f"--max-tokens {best['max_tokens']} "
            "--max-model-len 32768 "
            "--gpu-memory-utilization 0.90 "
            f"--max-num-seqs {best['max_num_seqs']} "
            f"--max-num-batched-tokens {best['max_num_batched_tokens']} "
            "--no-enable-prefix-caching "
            f"--generation-chunk-size {best['generation_chunk_size']} "
            "--retry-bad "
            f"--retry-k {best['retry_k']} "
            f"--retry-max-tokens {best['retry_max_tokens']}"
        )
    else:
        print("\nNo successful sweep runs yet.")


def build_run_command(
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    output_path: Path,
    raw_path: Path,
    metadata_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "run_inference.py",
        "--data-path",
        args.data_path,
        "--output-path",
        str(output_path),
        "--model-id",
        args.model_id,
        "--k",
        str(run_config["k"]),
        "--max-tokens",
        str(run_config["max_tokens"]),
        "--max-model-len",
        str(run_config["max_model_len"]),
        "--gpu-memory-utilization",
        str(run_config["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(run_config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(run_config["max_num_batched_tokens"]),
        "--generation-chunk-size",
        str(run_config["generation_chunk_size"]),
        "--retry-k",
        str(run_config["retry_k"]),
        "--retry-max-tokens",
        str(run_config["retry_max_tokens"]),
        "--raw-output-path",
        str(raw_path),
        "--metadata-path",
        str(metadata_path),
    ]
    cmd.append("--enable-chunked-prefill" if run_config["enable_chunked_prefill"] else "--no-enable-chunked-prefill")
    cmd.append("--enable-prefix-caching" if run_config["enable_prefix_caching"] else "--no-enable-prefix-caching")
    cmd.append("--retry-bad" if run_config["retry_bad"] else "--no-retry-bad")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.start_index:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.end_index is not None:
        cmd.extend(["--end-index", str(args.end_index)])
    return cmd


def write_failed_metadata(
    metadata_path: Path,
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    output_path: Path,
    raw_path: Path,
    returncode: int,
    elapsed_seconds: float,
) -> None:
    metadata = {
        "status": "failed",
        "error": f"run_inference.py exited with return code {returncode}",
        "returncode": returncode,
        "model_id": args.model_id,
        "data_path": args.data_path,
        "output_path": str(output_path),
        "raw_output_path": str(raw_path),
        "k": run_config["k"],
        "max_tokens": run_config["max_tokens"],
        "max_model_len": run_config["max_model_len"],
        "gpu_memory_utilization": run_config["gpu_memory_utilization"],
        "max_num_seqs": run_config["max_num_seqs"],
        "max_num_batched_tokens": run_config["max_num_batched_tokens"],
        "enable_chunked_prefill": run_config["enable_chunked_prefill"],
        "enable_prefix_caching": run_config["enable_prefix_caching"],
        "generation_chunk_size": run_config["generation_chunk_size"],
        "retry_bad": run_config["retry_bad"],
        "retry_k": run_config["retry_k"],
        "retry_max_tokens": run_config["retry_max_tokens"],
        "start_index": args.start_index,
        "end_index": args.end_index,
        "limit": args.limit,
        "elapsed_seconds": elapsed_seconds,
        "score_summary": None,
        "generation_summary": {},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}

    sweep_configs = SWEEP_PRESETS[args.preset]
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(sweep_configs):
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
            cmd = build_run_command(
                args=args,
                run_config=run_config,
                output_path=output_path,
                raw_path=raw_path,
                metadata_path=metadata_path,
            )
            print("Command:")
            print(" ".join(cmd))
            proc = subprocess.run(cmd)
            elapsed = time.time() - started
            if proc.returncode != 0:
                print(f"Run failed: {name} returncode={proc.returncode}")
                write_failed_metadata(
                    metadata_path,
                    args=args,
                    run_config=run_config,
                    output_path=output_path,
                    raw_path=raw_path,
                    returncode=proc.returncode,
                    elapsed_seconds=elapsed,
                )
                if not args.continue_on_failure:
                    raise SystemExit(proc.returncode)
            else:
                print(f"Finished {name} in {elapsed / 60:.2f} minutes")

        metadata = load_metadata(metadata_path)
        row = flatten_result(run_config, metadata)
        row["metadata_path"] = str(metadata_path)
        rows.append(row)
        write_summary(output_dir, rows)

        if args.cooldown_seconds > 0 and index < len(sweep_configs) - 1:
            print(f"\nCooling down for {args.cooldown_seconds:.1f} seconds before next candidate...")
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
