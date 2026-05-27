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
    "Give one answer per [ANS] in the order they appear, all inside a "
    "single \\boxed{} separated by commas (e.g. \\boxed{5, 7}). "
    "Put only the value in the box: no \"x =\", no units, no prose. "
    "If an [ANS] is followed by lettered choices, answer with the letter."
)

SYSTEM_PROMPT_MCQ = (
    "Output your final answer as a single capital letter from the given "
    "choices, inside \\boxed{} (e.g. \\boxed{C}). "
    "Do not put any formula, number, or option text in the box."
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


def load_raw_checkpoint(path: str | Path) -> dict[int, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    rows = load_jsonl(checkpoint_path)
    checkpoint: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "id" in row:
            checkpoint[int(row["id"])] = row
    return checkpoint


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


def should_retry(vote_info: dict[str, Any], samples: list[dict[str, Any]]) -> bool:
    if vote_info.get("vote_status") in {"all_none", "tie_first"}:
        return True
    return any(sample.get("finish_reason") == "length" for sample in samples)


def build_raw_rows(
    data: list[dict[str, Any]],
    per_question_samples: dict[int, list[dict[str, Any]]],
    retried_indices: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if idx not in per_question_samples:
            continue
        samples = per_question_samples[idx]
        sample_texts = [sample["text"] for sample in samples]
        _, vote_info = select_representative_response(item, sample_texts)
        rows.append({
            "id": item["id"],
            "is_mcq": bool(item.get("options")),
            "samples": samples,
            "retried": idx in retried_indices,
            **vote_info,
        })
    return rows


def score_public_if_available(
    data: list[dict[str, Any]],
    selected_responses: list[str],
) -> dict[str, Any] | None:
    if not data or "answer" not in data[0]:
        return None

    try:
        from judger import Judger

        judger = Judger(strict_extract=False)
    except Exception as exc:
        return {"error": f"Public scoring unavailable: {exc}"}

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


def compute_generation_summary(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(raw_rows)
    total_samples = 0
    total_tokens = 0
    boxed_any = 0
    boxed_all = 0
    retried = 0
    truncated_any = 0
    truncated_all = 0
    vote_status_counts: dict[str, int] = {}
    sample_count_counts: dict[str, int] = {}

    for row in raw_rows:
        samples = row.get("samples", [])
        total_samples += len(samples)
        total_tokens += sum(int(sample.get("n_tokens") or 0) for sample in samples)

        boxed_flags = [extract_boxed(sample.get("text")) is not None for sample in samples]
        if any(boxed_flags):
            boxed_any += 1
        if boxed_flags and all(boxed_flags):
            boxed_all += 1

        truncated_flags = [sample.get("finish_reason") == "length" for sample in samples]
        if any(truncated_flags):
            truncated_any += 1
        if truncated_flags and all(truncated_flags):
            truncated_all += 1

        if row.get("retried"):
            retried += 1

        status = str(row.get("vote_status") or "unknown")
        vote_status_counts[status] = vote_status_counts.get(status, 0) + 1

        sample_count_key = str(len(samples))
        sample_count_counts[sample_count_key] = sample_count_counts.get(sample_count_key, 0) + 1

    return {
        "num_questions": total,
        "total_samples": total_samples,
        "avg_samples_per_question": total_samples / total if total else 0.0,
        "avg_tokens_per_sample": total_tokens / total_samples if total_samples else 0.0,
        "boxed_any": {"count": boxed_any, "total": total, "rate": boxed_any / total if total else 0.0},
        "boxed_all": {"count": boxed_all, "total": total, "rate": boxed_all / total if total else 0.0},
        "truncated_any": {
            "count": truncated_any,
            "total": total,
            "rate": truncated_any / total if total else 0.0,
        },
        "truncated_all": {
            "count": truncated_all,
            "total": total,
            "rate": truncated_all / total if total else 0.0,
        },
        "retried": {"count": retried, "total": total, "rate": retried / total if total else 0.0},
        "vote_status_counts": vote_status_counts,
        "sample_count_counts": sample_count_counts,
    }


def _fmt_pct(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.2f}%"


def print_generation_summary(summary: dict[str, Any]) -> None:
    print("\n=== Generation Summary ===")
    print(f"Questions: {summary['num_questions']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Average samples/question: {summary['avg_samples_per_question']:.2f}")
    print(f"Average tokens/sample: {summary['avg_tokens_per_sample']:.2f}")

    boxed_any = summary["boxed_any"]
    boxed_all = summary["boxed_all"]
    truncated_any = summary["truncated_any"]
    truncated_all = summary["truncated_all"]
    retried = summary["retried"]

    print(
        "Boxed coverage any sample: "
        f"{boxed_any['count']} / {boxed_any['total']} ({_fmt_pct(boxed_any['rate'])})"
    )
    print(
        "Boxed coverage all samples: "
        f"{boxed_all['count']} / {boxed_all['total']} ({_fmt_pct(boxed_all['rate'])})"
    )
    print(
        "Truncated any sample: "
        f"{truncated_any['count']} / {truncated_any['total']} ({_fmt_pct(truncated_any['rate'])})"
    )
    print(
        "Truncated all samples: "
        f"{truncated_all['count']} / {truncated_all['total']} ({_fmt_pct(truncated_all['rate'])})"
    )
    print(f"Adaptive retry used: {retried['count']} / {retried['total']} ({_fmt_pct(retried['rate'])})")
    print(f"Vote statuses: {summary['vote_status_counts']}")
    print(f"Samples/question distribution: {summary['sample_count_counts']}")


def print_score_summary(score_summary: dict[str, Any] | None) -> None:
    if score_summary is None:
        print("\n=== Public Score ===")
        print("No public answers found in data; skipping accuracy.")
        return
    if score_summary.get("error"):
        print("\n=== Public Score ===")
        print(score_summary["error"])
        return

    print("\n=== Public Score ===")
    for label, key in [("Overall", "overall"), ("MCQ", "mcq"), ("Free-form", "free_form")]:
        row = score_summary[key]
        acc = row.get("accuracy")
        print(f"{label}: {row['correct']} / {row['total']} ({_fmt_pct(acc)})")


def selected_responses_from_raw_rows(
    data: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows_by_id = {int(row["id"]): row for row in raw_rows}
    completed_data: list[dict[str, Any]] = []
    selected_responses: list[str] = []
    for item in data:
        row = rows_by_id.get(int(item["id"]))
        if row is None:
            continue
        samples = row.get("samples", [])
        if not samples:
            continue
        sample_texts = [sample["text"] for sample in samples]
        selected, _ = select_representative_response(item, sample_texts)
        completed_data.append(item)
        selected_responses.append(selected)
    return completed_data, selected_responses


def print_checkpoint_summary(
    label: str,
    raw_rows: list[dict[str, Any]],
    data: list[dict[str, Any]],
) -> None:
    summary = compute_generation_summary(raw_rows)
    boxed_any = summary["boxed_any"]
    truncated_any = summary["truncated_any"]
    retried = summary["retried"]
    print(
        f"{label}: health "
        f"boxed_any={boxed_any['count']}/{boxed_any['total']} ({_fmt_pct(boxed_any['rate'])}), "
        f"truncated_any={truncated_any['count']}/{truncated_any['total']} ({_fmt_pct(truncated_any['rate'])}), "
        f"retried={retried['count']}/{retried['total']} ({_fmt_pct(retried['rate'])}), "
        f"vote_statuses={summary['vote_status_counts']}"
    )

    completed_data, selected_responses = selected_responses_from_raw_rows(data, raw_rows)
    score_summary = score_public_if_available(completed_data, selected_responses)
    if score_summary is not None:
        if score_summary.get("error"):
            print(f"{label}: partial public accuracy unavailable ({score_summary['error']})")
            return
        overall = score_summary["overall"]
        mcq = score_summary["mcq"]
        free_form = score_summary["free_form"]
        print(
            f"{label}: partial public accuracy "
            f"overall={overall['correct']}/{overall['total']} ({_fmt_pct(overall['accuracy'])}), "
            f"mcq={mcq['correct']}/{mcq['total']} ({_fmt_pct(mcq['accuracy'])}), "
            f"free={free_form['correct']}/{free_form['total']} ({_fmt_pct(free_form['accuracy'])})"
        )


def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "results/submission_final.csv",
    model_id: str = MODEL_ID,
    *,
    k: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.72,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    enable_thinking: bool = True,
    generation_chunk_size: int = 64,
    retry_bad: bool = True,
    retry_k: int = 2,
    retry_max_tokens: int = 4096,
    raw_output_path: str | None = None,
    metadata_path: str | None = None,
    limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    resume: bool = True,
) -> str:
    """Run the full end-to-end pipeline and write a Kaggle submission CSV.

    Defaults are tuned to fit one 24GB NVIDIA A30. `limit` is only for local
    smoke tests; leave it as None for the final private run.
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
    all_data = load_jsonl(data_path)
    if limit is not None:
        all_data = all_data[:limit]
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is None:
        end_index = len(all_data)
    if end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")
    data = all_data[start_index:end_index]
    if not data:
        raise ValueError(f"No rows loaded from {data_path}")
    if generation_chunk_size <= 0:
        raise ValueError("generation_chunk_size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_output_path is None:
        raw_output_path = str(out_path.with_suffix(".raw.jsonl"))
    raw_checkpoint_path = Path(raw_output_path)

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
        enable_prefix_caching=enable_prefix_caching,
    )
    sampling_params = SamplingParams(
        n=k,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    retry_sampling_params = SamplingParams(
        n=retry_k,
        max_tokens=retry_max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    per_question_samples: dict[int, list[dict[str, Any]]] = {}
    retried_indices: set[int] = set()

    if resume:
        checkpoint = load_raw_checkpoint(raw_checkpoint_path)
        if checkpoint:
            id_to_local_idx = {int(item["id"]): idx for idx, item in enumerate(data)}
            for qid, row in checkpoint.items():
                idx = id_to_local_idx.get(qid)
                if idx is None:
                    continue
                samples = row.get("samples") or []
                if samples:
                    per_question_samples[idx] = samples
                    if row.get("retried"):
                        retried_indices.add(idx)
            print(
                f"Resume: loaded {len(per_question_samples)} completed questions "
                f"from {raw_checkpoint_path}"
            )
            checkpoint_rows = build_raw_rows(data, per_question_samples, retried_indices)
            print_checkpoint_summary("Resume", checkpoint_rows, data)

    def save_checkpoint(label: str) -> None:
        checkpoint_rows = build_raw_rows(data, per_question_samples, retried_indices)
        write_jsonl(raw_checkpoint_path, checkpoint_rows)
        print(f"{label}: checkpointed {len(checkpoint_rows)} rows to {raw_checkpoint_path}")
        print_checkpoint_summary(label, checkpoint_rows, data)

    def generate_for_indices(
        indices: list[int],
        params: Any,
        *,
        label: str,
        append: bool = False,
    ) -> dict[int, list[dict[str, Any]]]:
        generated: dict[int, list[dict[str, Any]]] = {}
        total = len(indices)
        for start in range(0, total, generation_chunk_size):
            chunk_indices = indices[start:start + generation_chunk_size]
            chunk_prompts = [prompts[idx] for idx in chunk_indices]
            print(
                f"{label}: chunk {start // generation_chunk_size + 1} "
                f"({start + 1}-{start + len(chunk_indices)} / {total})"
            )
            chunk_outputs = llm.generate(chunk_prompts, sampling_params=params)
            for idx, output in zip(chunk_indices, chunk_outputs):
                samples = [
                    {
                        "text": sample.text.strip(),
                        "finish_reason": sample.finish_reason,
                        "n_tokens": len(sample.token_ids),
                    }
                    for sample in output.outputs
                ]
                generated[idx] = samples
                if append and idx in per_question_samples:
                    per_question_samples[idx].extend(samples)
                else:
                    per_question_samples[idx] = samples
            save_checkpoint(label)
        return generated

    print(f"Loaded {len(all_data)} total questions from {data_path}")
    print(f"Selected indices [{start_index}, {end_index}) -> {len(data)} questions")
    print(f"Generating with model={model_id}, k={k}, max_tokens={max_tokens}")
    missing_initial_indices = [
        idx for idx in range(len(data))
        if len(per_question_samples.get(idx, [])) < k
    ]
    if missing_initial_indices:
        generate_for_indices(
            missing_initial_indices,
            sampling_params,
            label="Initial generation",
        )
    else:
        print("Initial generation: all selected questions already completed.")

    retry_indices: list[int] = []
    if retry_bad and retry_k > 0:
        for idx, item in enumerate(data):
            if idx in retried_indices:
                continue
            sample_texts = [sample["text"] for sample in per_question_samples[idx]]
            _, vote_info = select_representative_response(item, sample_texts)
            if should_retry(vote_info, per_question_samples[idx]):
                retry_indices.append(idx)

    if retry_indices:
        print(
            f"Adaptive retry: regenerating {len(retry_indices)} low-confidence "
            f"questions with k={retry_k}, max_tokens={retry_max_tokens}"
        )
        retry_samples = generate_for_indices(
            retry_indices,
            retry_sampling_params,
            label="Adaptive retry",
            append=True,
        )
        retried_indices.update(retry_samples.keys())
        save_checkpoint("Adaptive retry")
    else:
        print("Adaptive retry: no low-confidence questions found.")

    submissions: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    selected_responses: list[str] = []

    for idx, item in enumerate(data):
        if idx not in per_question_samples:
            raise RuntimeError(f"Missing generated samples for local index {idx}, id={item.get('id')}")
        samples = per_question_samples[idx]
        sample_texts = [sample["text"] for sample in samples]
        selected, vote_info = select_representative_response(item, sample_texts)
        selected_responses.append(selected)
        submissions.append({"id": item["id"], "response": selected})
        raw_rows.append({
            "id": item["id"],
            "is_mcq": bool(item.get("options")),
            "samples": samples,
            "retried": idx in retried_indices,
            **vote_info,
        })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(submissions)

    write_jsonl(raw_output_path, raw_rows)

    elapsed = time.time() - start_time
    score_summary = score_public_if_available(data, selected_responses)
    generation_summary = compute_generation_summary(raw_rows)
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
        "retry_bad": retry_bad,
        "retry_k": retry_k,
        "retry_max_tokens": retry_max_tokens,
        "num_retried": len(retry_indices),
        "limit": limit,
        "start_index": start_index,
        "end_index": end_index,
        "resume": resume,
        "elapsed_seconds": elapsed,
        "generation_summary": generation_summary,
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
    print_generation_summary(generation_summary)
    print_score_summary(score_summary)

    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSE 151B final inference")
    parser.add_argument("--data-path", default="data/private.jsonl")
    parser.add_argument("--output-path", default="results/submission_final.csv")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-chunk-size", type=int, default=64)
    parser.add_argument("--retry-bad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-k", type=int, default=2)
    parser.add_argument("--retry-max-tokens", type=int, default=4096)
    parser.add_argument("--raw-output-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
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
        retry_bad=args.retry_bad,
        retry_k=args.retry_k,
        retry_max_tokens=args.retry_max_tokens,
        raw_output_path=args.raw_output_path,
        metadata_path=args.metadata_path,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
