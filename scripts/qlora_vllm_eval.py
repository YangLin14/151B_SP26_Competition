#!/usr/bin/env python3
"""Evaluate a base model or QLoRA adapter with vLLM."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


SYSTEM_PROMPT_FREEFORM = (
    "You are solving a mathematical free-response problem. "
    "Solve step by step. "
    "For every [ANS] placeholder, give the corresponding answer in the same order. "
    "Put all final answers inside one single \\boxed{} expression. "
    "If there are multiple answers, separate them by commas, for example \\boxed{3, 7}. "
    "Do not include units unless the problem explicitly asks for units."
)

SYSTEM_PROMPT_MCQ = (
    "You are solving a mathematical multiple-choice problem. "
    "Solve the problem carefully, then compare your result with the answer choices. "
    "The final answer must be exactly one capital letter inside \\boxed{}, such as \\boxed{C}. "
    "Do not put the numerical value or explanation inside the final box."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QLoRA adapter with vLLM")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--adapter-name", default="qlora_adapter")
    parser.add_argument("--data-path", default="data/public.jsonl")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(item: dict[str, Any]) -> tuple[str, str]:
    question = str(item["question"]).strip()
    options = item.get("options")

    if options:
        if isinstance(options, dict):
            option_lines = [f"{key}. {value}" for key, value in sorted(options.items())]
        else:
            option_lines = [
                f"{chr(ord('A') + idx)}. {value}" for idx, value in enumerate(options)
            ]
        user = (
            f"{question}\n\nOptions:\n"
            + "\n".join(option_lines)
            + "\n\nSolve the problem and end with the required boxed letter."
        )
        return SYSTEM_PROMPT_MCQ, user

    user = f"{question}\n\nSolve the problem and end with the required boxed answer."
    return SYSTEM_PROMPT_FREEFORM, user


def extract_boxed(text: str | None) -> str | None:
    if text is None:
        return None

    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    chars = []

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


def get_allowed_letters(item: dict[str, Any]) -> str:
    options = item.get("options")
    if isinstance(options, dict):
        keys = [str(k).strip().upper() for k in options.keys()]
        letter_keys = [k for k in keys if len(k) == 1 and k in string.ascii_uppercase]
        if letter_keys:
            return "".join(letter_keys)
        return string.ascii_uppercase[: len(options)]
    if isinstance(options, list):
        return string.ascii_uppercase[: len(options)]
    return string.ascii_uppercase[:10]


def extract_letter(text: str | None, allowed: str | None = None) -> str | None:
    if text is None:
        return None
    if allowed is None:
        allowed = string.ascii_uppercase[:10]
    allowed = "".join(c for c in allowed.upper() if c in string.ascii_uppercase)
    char_class = re.escape(allowed)

    s = str(text).strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "")

    boxed_matches = re.findall(
        rf"\\boxed\s*\{{\s*\(?\s*([{char_class}])\s*\)?\s*\}}",
        s,
        flags=re.IGNORECASE,
    )
    if boxed_matches:
        return boxed_matches[-1].upper()

    upper_s = s.upper()
    patterns = [
        rf"(?:THEREFORE|THUS|SO|FINAL ANSWER|ANSWER|THE ANSWER IS)\s*(?:IS|:)?\s*\(?\s*([{char_class}])\s*\)?",
        rf"\bOPTION\s+([{char_class}])\b",
        rf"\bCHOICE\s+([{char_class}])\b",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, upper_s):
            candidates.append((match.start(), match.group(1).upper()))
    if candidates:
        return sorted(candidates, key=lambda x: x[0])[-1][1]

    match = re.fullmatch(rf"\s*\(?\s*([{char_class}])\s*\)?\.?\s*", upper_s)
    if match:
        return match.group(1).upper()
    return None


def majority_vote(values: list[str | None]) -> tuple[str | None, str]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None, "all_none"

    counts = {}
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


def score_results(eval_data: list[dict[str, Any]], per_question_raw: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    from judger import Judger

    judger = Judger(strict_extract=False)
    results = []

    for item, samples in zip(eval_data, per_question_raw):
        is_mcq = bool(item.get("options"))
        gold = item.get("answer")
        sample_texts = [sample["text"] for sample in samples]
        sample_boxed = [extract_boxed(text) for text in sample_texts]
        voted, vote_status = majority_vote(sample_boxed)
        rep_idx = next((idx for idx, boxed in enumerate(sample_boxed) if boxed == voted), 0)
        rep_text = sample_texts[rep_idx]

        if is_mcq:
            pred_letter = extract_letter(voted, allowed=get_allowed_letters(item))
            if pred_letter is None:
                pred_letter = extract_letter(rep_text, allowed=get_allowed_letters(item))
            correct = pred_letter == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            pred_for_judge = f"\\boxed{{{voted}}}" if voted is not None else rep_text
            pred_letter = None
            try:
                correct = judger.auto_judge(
                    pred=pred_for_judge,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append(
            {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "K": len(samples),
                "samples_boxed": sample_boxed,
                "voted": voted,
                "vote_status": vote_status,
                "rep_response": rep_text,
                "pred_letter": pred_letter,
                "correct": correct,
                "tokens_per_sample": [sample["n_tokens"] for sample in samples],
                "finish_reasons": [sample["finish_reason"] for sample in samples],
            }
        )

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    def summarize(name: str, subset: list[dict[str, Any]]) -> None:
        if not subset:
            return
        correct = sum(bool(row["correct"]) for row in subset)
        print(f"{name:10s}: {correct:4d} / {len(subset):4d} ({correct / len(subset) * 100:.2f}%)")

    summarize("MCQ", [row for row in results if row["is_mcq"]])
    summarize("Free-form", [row for row in results if not row["is_mcq"]])
    summarize("Overall", results)


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from vllm import LLM, SamplingParams

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        @property
        def _all_special_tokens_extended(self: Any) -> list[str]:
            return list(self.all_special_tokens)

        Qwen2Tokenizer.all_special_tokens_extended = _all_special_tokens_extended

    data = load_jsonl(Path(args.data_path))
    eval_data = data if args.n_eval is None or args.n_eval < 0 else data[: args.n_eval]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    prompts = []
    for item in eval_data:
        system, user = build_prompt(item)
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    enable_lora = args.adapter_path is not None
    llm_kwargs = {
        "model": args.model_id,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": True,
    }
    lora_request = None
    if enable_lora:
        from vllm.lora.request import LoRARequest

        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest(
            lora_name=args.adapter_name,
            lora_int_id=1,
            lora_path=args.adapter_path,
        )

    model = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=args.k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    outputs = model.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    per_question_raw = []
    for output in outputs:
        samples = []
        for sample in output.outputs:
            samples.append(
                {
                    "text": sample.text,
                    "finish_reason": sample.finish_reason,
                    "n_tokens": len(sample.token_ids),
                }
            )
        per_question_raw.append(samples)

    results = score_results(eval_data, per_question_raw)
    print_summary(results)

    if args.output_path is None:
        suffix = "adapter" if enable_lora else "base"
        output_path = Path(f"results/qlora_eval_{suffix}_{len(eval_data)}.jsonl")
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "n_eval": len(eval_data),
        "k": args.k,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_lora_rank": args.max_lora_rank if enable_lora else None,
        "output_path": str(output_path),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
