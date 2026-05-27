#!/usr/bin/env python3
"""Single-entry competition inference pipeline.

This file intentionally performs pure model inference only. It does not execute
model-generated code, call calculators, or use external APIs at test time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT_FREEFORM = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "First count the [ANS] placeholders in the problem. If the problem has "
    "multiple placeholders, give exactly that many final answers in the same "
    "order. Put all final answers inside one single \\boxed{} expression, "
    "separated by commas, for example \\boxed{3, 7}. Do not include units "
    "unless the problem explicitly asks for units."
)

SYSTEM_PROMPT_MCQ = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "Solve the problem carefully, then compare your result with the answer "
    "choices. Your boxed answer must contain exactly one capital letter from "
    "the provided choices, for example \\boxed{C}. Do not put the numerical "
    "value, option text, or explanation inside the final box."
)


def configure_environment() -> None:
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


def build_prompt(item: dict[str, Any]) -> tuple[str, str]:
    question = str(item["question"]).strip()
    options = item.get("options")

    if options:
        labels = [chr(ord("A") + idx) for idx in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {str(option).strip()}"
            for label, option in zip(labels, options)
        )
        user = (
            f"Problem:\n{question}\n\n"
            f"Answer choices:\n{opts_text}\n\n"
            "Solve the problem and end with the required boxed letter."
        )
        return SYSTEM_PROMPT_MCQ, user

    ans_count = question.count("[ANS]")
    count_hint = (
        f"\n\nThis problem contains {ans_count} [ANS] placeholder"
        f"{'' if ans_count == 1 else 's'}."
        if ans_count
        else ""
    )
    user = (
        f"Problem:\n{question}{count_hint}\n\n"
        "Solve the problem and end with the required boxed answer."
    )
    return SYSTEM_PROMPT_FREEFORM, user


def render_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_boxed(text: str | None) -> str | None:
    """Extract the final nested \\boxed{...} content from a model response."""
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


def allowed_letters(item: dict[str, Any]) -> str:
    options = item.get("options")
    if isinstance(options, list):
        return string.ascii_uppercase[: len(options)]
    return string.ascii_uppercase[:10]


def extract_letter(text: str | None, allowed: str) -> str | None:
    if not text:
        return None

    allowed = "".join(c for c in allowed.upper() if c in string.ascii_uppercase)
    char_class = re.escape(allowed)
    s = str(text)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "")

    boxed = extract_boxed(s)
    if boxed:
        m = re.fullmatch(rf"\s*\(?\s*([{char_class}])\s*\)?\.?\s*", boxed, re.I)
        if m:
            return m.group(1).upper()

    upper_s = s.upper()
    patterns = [
        rf"(?:FINAL ANSWER|ANSWER|THE ANSWER IS|THEREFORE|THUS|SO)\s*(?:IS|:)?\s*\(?\s*([{char_class}])\s*\)?",
        rf"\b(?:OPTION|CHOICE)\s+([{char_class}])\b",
    ]
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, upper_s):
            candidates.append((match.start(), match.group(1).upper()))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    return None


def normalize_vote_value(value: str | None, *, is_mcq: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .")
    return text.upper() if is_mcq else text.lower()


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


def select_representative_response(
    item: dict[str, Any],
    sample_texts: list[str],
) -> tuple[str, dict[str, Any]]:
    is_mcq = bool(item.get("options"))

    if is_mcq:
        letters = [extract_letter(text, allowed_letters(item)) for text in sample_texts]
        vote_values = [normalize_vote_value(letter, is_mcq=True) for letter in letters]
    else:
        boxed = [extract_boxed(text) for text in sample_texts]
        vote_values = [normalize_vote_value(value, is_mcq=False) for value in boxed]

    voted, vote_status = majority_vote(vote_values)
    rep_idx = 0
    if voted is not None:
        for idx, value in enumerate(vote_values):
            if value == voted:
                rep_idx = idx
                break

    return sample_texts[rep_idx], {
        "vote_values": vote_values,
        "voted": voted,
        "vote_status": vote_status,
        "rep_idx": rep_idx,
    }


def score_public_if_available(
    data: list[dict[str, Any]],
    selected_responses: list[str],
) -> dict[str, Any] | None:
    if not data or "answer" not in data[0]:
        return None

    from judger import Judger

    judger = Judger(strict_extract=False)
    correct = 0
    mcq_total = mcq_correct = 0
    free_total = free_correct = 0

    for item, response in zip(data, selected_responses):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]
        if is_mcq:
            pred = extract_letter(response, allowed_letters(item))
            ok = pred == str(gold).strip().upper()
            mcq_total += 1
            mcq_correct += int(ok)
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                ok = bool(judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                ))
            except Exception:
                ok = False
            free_total += 1
            free_correct += int(ok)
        correct += int(ok)

    return {
        "overall": {"correct": correct, "total": len(data), "accuracy": correct / len(data)},
        "mcq": {
            "correct": mcq_correct,
            "total": mcq_total,
            "accuracy": mcq_correct / mcq_total if mcq_total else None,
        },
        "free_form": {
            "correct": free_correct,
            "total": free_total,
            "accuracy": free_correct / free_total if free_total else None,
        },
    }


def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/submission_final.csv",
    model_id: str = MODEL_ID,
    *,
    k: int = 5,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 32768,
    enable_chunked_prefill: bool = True,
    enable_thinking: bool = True,
    raw_output_path: str | None = None,
    metadata_path: str | None = None,
    limit: int | None = None,
) -> str:
    """Run the full end-to-end pipeline and write a Kaggle submission CSV.

    Defaults are tuned for one NVIDIA A30. `limit` is only for local smoke tests;
    leave it as None for the final private run.
    """
    configure_environment()

    from transformers import AutoTokenizer
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from vllm import LLM, SamplingParams

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        @property
        def _all_special_tokens_extended(self: Any) -> list[str]:
            return list(self.all_special_tokens)

        Qwen2Tokenizer.all_special_tokens_extended = _all_special_tokens_extended

    start_time = time.time()
    data = load_jsonl(data_path)
    if limit is not None:
        data = data[:limit]
    if not data:
        raise ValueError(f"No rows loaded from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for item in data:
        system_prompt, user_prompt = build_prompt(item)
        prompts.append(
            render_chat_prompt(
                tokenizer,
                system_prompt,
                user_prompt,
                enable_thinking=enable_thinking,
            )
        )

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    print(f"Loaded {len(data)} questions from {data_path}")
    print(f"Generating with model={model_id}, k={k}, max_tokens={max_tokens}")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    submissions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    selected_responses: list[str] = []

    for item, output in zip(data, outputs):
        sample_texts = [sample.text.strip() for sample in output.outputs]
        selected, vote_info = select_representative_response(item, sample_texts)
        selected_responses.append(selected)
        submissions.append({"id": item["id"], "response": selected})
        raw_rows.append({
            "id": item["id"],
            "is_mcq": bool(item.get("options")),
            "samples": [
                {
                    "text": sample.text.strip(),
                    "finish_reason": sample.finish_reason,
                    "n_tokens": len(sample.token_ids),
                }
                for sample in output.outputs
            ],
            **vote_info,
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(submissions)

    if raw_output_path is None:
        raw_output_path = str(out_path.with_suffix(".raw.jsonl"))
    write_jsonl(raw_output_path, raw_rows)

    elapsed = time.time() - start_time
    score_summary = score_public_if_available(data, selected_responses)
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
        "enable_thinking": enable_thinking,
        "limit": limit,
        "elapsed_seconds": elapsed,
        "score_summary": score_summary,
    }
    if metadata_path is None:
        metadata_path = str(out_path.with_suffix(".metadata.json"))
    Path(metadata_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved submission CSV: {out_path}")
    print(f"Saved raw generations: {raw_output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    if score_summary is not None:
        overall = score_summary["overall"]
        print(
            "Public score: "
            f"{overall['correct']} / {overall['total']} "
            f"({overall['accuracy'] * 100:.2f}%)"
        )

    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSE 151B final inference")
    parser.add_argument("--data-path", default="data/private.jsonl")
    parser.add_argument("--output-path", default="results/submission_final.csv")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-output-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
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
        enable_thinking=args.enable_thinking,
        raw_output_path=args.raw_output_path,
        metadata_path=args.metadata_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
