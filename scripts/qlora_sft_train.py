#!/usr/bin/env python3
"""Train a QLoRA adapter for the CSE 151B math competition.

This script is intentionally separate from the notebooks so QLoRA runs are
repeatable and easy to scale from a small smoke test to a full training run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math solver. "
    "Solve the problem step by step, keeping the reasoning concise. "
    "Put the final answer in \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT training")

    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--run-name", default="qlora_sft_public_smoke")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--data-source",
        choices=["public", "numina"],
        default="public",
        help=(
            "public uses data/public.jsonl with an internal train/dev split; "
            "numina uses AI-MO/NuminaMath-CoT from Hugging Face."
        ),
    )
    parser.add_argument("--public-data-path", default="data/public.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--numina-dataset", default="AI-MO/NuminaMath-CoT")
    parser.add_argument(
        "--numina-streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream NuminaMath examples instead of materializing the full 859k-example train split.",
    )
    parser.add_argument(
        "--numina-shuffle-buffer",
        type=int,
        default=10_000,
        help="Shuffle buffer used when --numina-streaming is enabled.",
    )
    parser.add_argument("--max-train-examples", type=int, default=200)

    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Use a small positive number for smoke tests; use -1 for full epoch.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when formatting training conversations.",
    )

    return parser.parse_args()


def configure_environment() -> None:
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def public_answer_to_boxed(answer: Any) -> str:
    if isinstance(answer, list):
        content = ", ".join(str(x) for x in answer)
    else:
        content = str(answer)
    return f"\\boxed{{{content}}}"


def format_public_user(item: dict[str, Any]) -> str:
    question = str(item["question"]).strip()
    options = item.get("options")
    if not options:
        return question

    if isinstance(options, dict):
        option_lines = [f"{key}. {value}" for key, value in sorted(options.items())]
    else:
        option_lines = []
        for idx, value in enumerate(options):
            option_lines.append(f"{chr(ord('A') + idx)}. {value}")

    return question + "\n\nOptions:\n" + "\n".join(option_lines)


def build_public_datasets(
    *,
    path: Path,
    train_ratio: float,
    max_train_examples: int,
    seed: int,
    output_dir: Path,
    tokenizer: Any,
    system_prompt: str,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import Dataset

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    rows = load_jsonl(path)
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_rows = shuffled[:split_idx]
    dev_rows = shuffled[split_idx:]

    if max_train_examples > 0:
        train_rows = train_rows[:max_train_examples]

    def format_item(item: dict[str, Any]) -> dict[str, str]:
        target = (
            "We need solve the problem and provide the final answer.\n"
            f"{public_answer_to_boxed(item['answer'])}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_public_user(item)},
            {"role": "assistant", "content": target},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "public_train_split.jsonl", train_rows)
    write_jsonl(output_dir / "public_dev_split.jsonl", dev_rows)

    return Dataset.from_list([format_item(row) for row in train_rows]), train_rows, dev_rows


def build_numina_dataset(
    *,
    dataset_name: str,
    max_train_examples: int,
    seed: int,
    output_dir: Path,
    tokenizer: Any,
    system_prompt: str,
    streaming: bool,
    shuffle_buffer: int,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import Dataset, load_dataset

    if streaming:
        if max_train_examples <= 0:
            raise ValueError(
                "Numina streaming requires --max-train-examples > 0. "
                "Use a bounded subset such as 500, 5000, or pass --no-numina-streaming for full materialization."
            )

        print(
            f"Streaming {max_train_examples} examples from {dataset_name} "
            f"with shuffle_buffer={shuffle_buffer}..."
        )
        stream = load_dataset(dataset_name, split="train", streaming=True)
        if shuffle_buffer > 1:
            stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
        raw_rows = list(stream.take(max_train_examples))
    else:
        print(f"Loading full {dataset_name} train split before subsetting...")
        raw_train = load_dataset(dataset_name, split="train")
        if max_train_examples > 0 and max_train_examples < len(raw_train):
            raw_train = raw_train.shuffle(seed=seed).select(range(max_train_examples))
        raw_rows = list(raw_train)

    def format_item(example: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "numina_train_subset.jsonl", raw_rows)

    formatted = Dataset.from_list([format_item(row) for row in raw_rows])
    return formatted, raw_rows, []


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the installed library version."""
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def main() -> None:
    args = parse_args()
    configure_environment()

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. QLoRA training should run on GPU.")

    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        @property
        def _all_special_tokens_extended(self: Any) -> list[str]:
            return list(self.all_special_tokens)

        Qwen2Tokenizer.all_special_tokens_extended = _all_special_tokens_extended

    output_dir = Path(args.output_root) / args.run_name
    adapter_dir = output_dir / "final_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Run:", args.run_name)
    print("Model:", args.model_id)
    print("Output:", output_dir)
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.data_source == "public":
        train_dataset, train_rows, dev_rows = build_public_datasets(
            path=Path(args.public_data_path),
            train_ratio=args.train_ratio,
            max_train_examples=args.max_train_examples,
            seed=args.seed,
            output_dir=output_dir,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
        )
    else:
        train_dataset, train_rows, dev_rows = build_numina_dataset(
            dataset_name=args.numina_dataset,
            max_train_examples=args.max_train_examples,
            seed=args.seed,
            output_dir=output_dir,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            streaming=args.numina_streaming,
            shuffle_buffer=args.numina_shuffle_buffer,
        )

    print("Training examples:", len(train_dataset))
    print("Sample training text:")
    print(train_dataset[0]["text"][:1200])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config_params = inspect.signature(SFTConfig).parameters
    length_key = "max_seq_length" if "max_seq_length" in sft_config_params else "max_length"
    if length_key not in sft_config_params:
        raise RuntimeError("Installed TRL SFTConfig supports neither max_seq_length nor max_length.")

    sft_config_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        length_key: args.max_seq_len,
        "packing": False,
        "dataset_text_field": "text",
        "report_to": "none",
        "seed": args.seed,
    }
    training_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_config_kwargs))

    metadata = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "data_source": args.data_source,
        "public_data_path": args.public_data_path if args.data_source == "public" else None,
        "numina_dataset": args.numina_dataset if args.data_source == "numina" else None,
        "numina_streaming": args.numina_streaming if args.data_source == "numina" else None,
        "numina_shuffle_buffer": args.numina_shuffle_buffer if args.data_source == "numina" else None,
        "num_train_examples": len(train_dataset),
        "num_public_dev_examples": len(dev_rows),
        "train_ratio": args.train_ratio if args.data_source == "public" else None,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "adapter_dir": str(adapter_dir),
    }
    save_metadata(output_dir / "run_metadata.pretrain.json", metadata)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer, trainer_kwargs))

    t0 = time.time()
    train_result = trainer.train()
    train_minutes = (time.time() - t0) / 60

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata["train_runtime_minutes"] = train_minutes
    metadata["train_metrics"] = train_result.metrics
    save_metadata(output_dir / "run_metadata.json", metadata)

    print(f"Training complete in {train_minutes:.2f} min")
    print(f"Adapter saved to {adapter_dir}")
    print(f"Metadata saved to {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
