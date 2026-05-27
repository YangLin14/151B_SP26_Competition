#!/usr/bin/env python3
"""Evaluate a base model or QLoRA adapter with Transformers.

This is the Windows-native evaluation path. It avoids vLLM, which is not a good
fit for native Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QLoRA adapter with Transformers")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--data-path", default="data/public.jsonl")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model with bitsandbytes 4-bit quantization. Use --no-load-in-4bit on unstable H100 MIG nodes.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation for Transformers eval.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=True to Qwen chat template when supported.",
    )
    parser.add_argument("--tracker-path", default="docs/QLORA_RESULTS_TRACKER.md")
    parser.add_argument("--tracker-eval-id", default=None)
    parser.add_argument("--tracker-notes", default="")
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the Markdown QLoRA results tracker after eval.",
    )
    return parser.parse_args()


def configure_environment() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    configure_environment()

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(script_dir))
    from qlora_update_tracker import update_tracker
    from qlora_vllm_eval import build_prompt, print_summary, render_chat_prompt, score_results

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Transformers eval should run on GPU.")

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        @property
        def _all_special_tokens_extended(self: Any) -> list[str]:
            return list(self.all_special_tokens)

        Qwen2Tokenizer.all_special_tokens_extended = _all_special_tokens_extended

    data = load_jsonl(Path(args.data_path))
    eval_data = data if args.n_eval is None or args.n_eval < 0 else data[: args.n_eval]

    if args.output_path is None:
        suffix = "adapter" if args.adapter_path else "base"
        output_path = Path(f"results/qlora_transformers_eval_{suffix}_{len(eval_data)}.jsonl")
    else:
        output_path = Path(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for item in eval_data:
        system, user = build_prompt(item)
        prompts.append(
            render_chat_prompt(
                tokenizer,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                enable_thinking=args.enable_thinking,
            )
        )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print("Loading model:", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs,
    )

    if args.adapter_path:
        print("Loading adapter:", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    do_sample = args.temperature > 0
    per_question_raw = []

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                top_k=args.top_k if do_sample else None,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        samples = []
        for seq in output_ids:
            new_tokens = seq[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            samples.append(
                {
                    "text": text,
                    "finish_reason": "unknown",
                    "n_tokens": int(new_tokens.numel()),
                }
            )
        per_question_raw.append(samples)

    raw_path = output_path.with_suffix(".raw.jsonl")
    write_jsonl(
        raw_path,
        [
            {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "gold": item.get("answer"),
                "samples": samples,
            }
            for item, samples in zip(eval_data, per_question_raw)
        ],
    )
    print(f"Saved raw generations to {raw_path}")

    results = score_results(eval_data, per_question_raw)
    print_summary(results)

    write_jsonl(output_path, results)

    metadata = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "n_eval": len(eval_data),
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "max_input_length": args.max_input_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "enable_thinking": args.enable_thinking,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation": args.attn_implementation,
        "output_path": str(output_path),
        "eval_backend": "transformers",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {output_path}")

    if args.update_tracker:
        try:
            update_tracker(
                tracker_path=Path(args.tracker_path),
                results_path=output_path,
                metadata_path=output_path.with_suffix(".metadata.json"),
                eval_id=args.tracker_eval_id,
                notes=args.tracker_notes,
            )
            print(f"Updated tracker: {args.tracker_path}")
        except Exception as exc:
            print(f"Warning: failed to update tracker: {exc}")


if __name__ == "__main__":
    main()
