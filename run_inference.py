#!/usr/bin/env python3
"""Single-entry K1 private inference pipeline for the CSE 151B competition."""

from __future__ import annotations

import argparse
import csv
import json
import os
import string
import time
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT_FREEFORM = """Please Give one answer per [ANS] in the order they appear, all inside a single \\boxed{} separated by commas (e.g. \\boxed{5, 7}).
Put only the value in the box: no "x =", no units, no prose.
If an [ANS] is followed by lettered choices, answer with the letter.""".strip()

SYSTEM_PROMPT_MCQ = """Please Output your final answer as a single capital letter from the given choices, inside \\boxed{} (e.g. \\boxed{C}).
Do not put any formula, number, or option text in the box.""".strip()


def configure_environment() -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(question: str, options: list[Any] | None = None) -> tuple[str, str]:
    if options:
        labels = [string.ascii_uppercase[i] for i in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {str(option).strip()}"
            for label, option in zip(labels, options)
        )
        user_prompt = (
            f"Problem:\n{question}\n\n"
            f"Answer choices:\n{opts_text}\n\n"
            "Solve the problem and end with the required boxed letter."
        )
        return SYSTEM_PROMPT_MCQ, user_prompt

    user_prompt = (
        f"Problem:\n{question}\n\n"
        "Solve the problem and end with the required boxed answer."
    )
    return SYSTEM_PROMPT_FREEFORM, user_prompt


def render_chat_prompt(tokenizer: Any, item: dict[str, Any], *, enable_thinking: bool) -> str:
    system_prompt, user_prompt = build_prompt(item["question"], item.get("options"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_boxed(text: str | None) -> str | None:
    if not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    chars: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None


def majority_vote(values: list[str | None]) -> tuple[str | None, str]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None, "all_none"

    counts: dict[str, int] = {}
    for value in valid:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == max_count}
    if len(winners) == 1:
        return next(iter(winners)), "majority"

    for value in valid:
        if value in winners:
            return value, "tie_first"
    return None, "all_none"


def select_response(sample_texts: list[str]) -> tuple[str, dict[str, Any]]:
    boxed_answers = [extract_boxed(text) for text in sample_texts]
    voted, vote_status = majority_vote(boxed_answers)
    rep_idx = next((idx for idx, boxed in enumerate(boxed_answers) if boxed == voted), 0)
    return sample_texts[rep_idx], {
        "boxed_answers": boxed_answers,
        "voted": voted,
        "vote_status": vote_status,
        "rep_idx": rep_idx,
    }


def load_raw_checkpoint(path: str | Path) -> dict[int, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    rows = load_jsonl(checkpoint_path)
    return {int(row["id"]): row for row in rows if "id" in row and row.get("samples")}


def patch_qwen_tokenizer() -> None:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    if hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self: Any) -> list[str]:
        return list(self.all_special_tokens)

    Qwen2Tokenizer.all_special_tokens_extended = all_special_tokens_extended


def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/submission_final.csv",
    model_id: str = MODEL_ID,
    *,
    k: int = 1,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.92,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 16384,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = True,
    enable_thinking: bool = True,
    generation_chunk_size: int = 64,
    raw_output_path: str | None = None,
    metadata_path: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> str:
    """Run the full K1 model pipeline and write the final submission CSV.

    The defaults match the private K1 notebook used for the submitted result:
    Qwen3-4B-Thinking, one sample per question, 16K generated-token budget,
    thinking mode enabled, and A30-oriented vLLM settings.
    """
    configure_environment()
    if generation_chunk_size <= 0:
        raise ValueError("generation_chunk_size must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    patch_qwen_tokenizer()

    started = time.time()
    data = load_jsonl(data_path)
    if limit is not None:
        data = data[:limit]
    if not data:
        raise ValueError(f"No rows loaded from {data_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_output_path is None:
        raw_output_path = str(out_path.with_suffix(".raw.jsonl"))
    if metadata_path is None:
        metadata_path = str(out_path.with_suffix(".metadata.json"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        render_chat_prompt(tokenizer, item, enable_thinking=enable_thinking)
        for item in data
    ]

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=k,
        repetition_penalty=repetition_penalty,
    )

    rows_by_id = load_raw_checkpoint(raw_output_path) if resume else {}
    id_to_index = {int(item["id"]): idx for idx, item in enumerate(data)}
    rows_by_id = {
        qid: row for qid, row in rows_by_id.items()
        if qid in id_to_index and len(row.get("samples", [])) >= k
    }
    if rows_by_id:
        print(f"Resume: loaded {len(rows_by_id)} completed rows from {raw_output_path}")

    def checkpoint() -> None:
        ordered = [
            rows_by_id[int(item["id"])]
            for item in data
            if int(item["id"]) in rows_by_id
        ]
        write_jsonl(raw_output_path, ordered)

    missing_indices = [
        idx for idx, item in enumerate(data)
        if int(item["id"]) not in rows_by_id
    ]
    print(f"Loaded {len(data)} questions from {data_path}")
    print(f"Generating {len(missing_indices)} missing questions with model={model_id}")
    print(
        "Hyperparameters: "
        f"k={k}, max_tokens={max_tokens}, temperature={temperature}, "
        f"top_p={top_p}, top_k={top_k}"
    )

    for start in range(0, len(missing_indices), generation_chunk_size):
        chunk_indices = missing_indices[start:start + generation_chunk_size]
        chunk_prompts = [prompts[idx] for idx in chunk_indices]
        print(
            f"Generation chunk {start // generation_chunk_size + 1}: "
            f"{start + 1}-{start + len(chunk_indices)} / {len(missing_indices)}"
        )
        outputs = llm.generate(chunk_prompts, sampling_params=sampling_params)
        for idx, output in zip(chunk_indices, outputs):
            samples = [
                {
                    "text": sample.text.strip(),
                    "finish_reason": sample.finish_reason,
                    "n_tokens": len(sample.token_ids),
                }
                for sample in output.outputs
            ]
            sample_texts = [sample["text"] for sample in samples]
            _, vote_info = select_response(sample_texts)
            item = data[idx]
            rows_by_id[int(item["id"])] = {
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "samples": samples,
                **vote_info,
            }
        checkpoint()

    submission_rows: list[dict[str, str | int]] = []
    raw_rows: list[dict[str, Any]] = []
    for item in data:
        row = rows_by_id[int(item["id"])]
        sample_texts = [sample["text"] for sample in row["samples"]]
        selected, vote_info = select_response(sample_texts)
        row.update(vote_info)
        submission_rows.append({"id": item["id"], "response": selected})
        raw_rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(submission_rows)
    write_jsonl(raw_output_path, raw_rows)

    elapsed = time.time() - started
    metadata = {
        "model_id": model_id,
        "data_path": data_path,
        "output_path": str(out_path),
        "raw_output_path": raw_output_path,
        "num_questions": len(data),
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "enable_thinking": enable_thinking,
        "generation_chunk_size": generation_chunk_size,
        "limit": limit,
        "resume": resume,
        "elapsed_seconds": elapsed,
        "vote_status_counts": {
            status: sum(1 for row in raw_rows if row.get("vote_status") == status)
            for status in sorted({str(row.get("vote_status")) for row in raw_rows})
        },
    }
    Path(metadata_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved submission CSV: {out_path}")
    print(f"Saved raw generations: {raw_output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final K1 private inference")
    parser.add_argument("--data-path", default="data/private.jsonl")
    parser.add_argument("--output-path", default="results/submission_final.csv")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-chunk-size", type=int, default=64)
    parser.add_argument("--raw-output-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        data_path=args.data_path,
        output_path=args.output_path,
        model_id=args.model_id,
        k=args.k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_thinking=args.enable_thinking,
        generation_chunk_size=args.generation_chunk_size,
        raw_output_path=args.raw_output_path,
        metadata_path=args.metadata_path,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
