# QLoRA workflow

This workflow is for building a competition-use LoRA adapter without depending
on notebook state.

## 1. Smoke test on the public split

Use this first. It checks that model loading, 4-bit quantization, LoRA attachment,
dataset formatting, training, saving, and metadata writing all work.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_smoke \
  --data-source public \
  --max-train-examples 200 \
  --max-steps 50 \
  --max-seq-len 2048
```

Expected outputs:

```text
outputs/qlora_sft_public_smoke/final_adapter/
outputs/qlora_sft_public_smoke/public_train_split.jsonl
outputs/qlora_sft_public_smoke/public_dev_split.jsonl
outputs/qlora_sft_public_smoke/run_metadata.json
```

Use the saved `public_dev_split.jsonl` as the first adapter validation set. Do
not report train-split accuracy as model quality.

## 2. Larger public-split run

After the smoke test is stable, train on more of the public split.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_v1 \
  --data-source public \
  --max-train-examples -1 \
  --max-steps -1 \
  --max-seq-len 2048
```

This is useful for learning the competition output format, but it does not give
strong reasoning supervision because the public data only has final answers.

## 3. CoT SFT run

For a more realistic QLoRA experiment, use a dataset with worked solutions.
The training script streams NuminaMath by default and materializes only the
requested subset into memory, then saves that subset under the run output
directory.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_numina_5k_v1 \
  --data-source numina \
  --max-train-examples 5000 \
  --max-steps -1 \
  --max-seq-len 4096 \
  --numina-shuffle-buffer 10000
```

This requires Hugging Face dataset access on the training machine.

## 4. Evaluate Base And Adapter

Use Transformers for fair base-vs-adapter comparison. In our current Qwen3 +
vLLM environment, vLLM falls back to a Transformers backend that does not support
LoRA adapters, so vLLM is base-only.

Base control:

```bash
python scripts/qlora_transformers_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --enable-thinking \
  --output-path results/qlora_base_control_transformers_eval_50.jsonl \
  --tracker-eval-id base_tf_same_as_numina_5k_50
```

Adapter eval:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_numina_5k_2048/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --enable-thinking \
  --output-path results/qlora_numina_5k_2048_transformers_eval_50.jsonl \
  --tracker-eval-id numina_5k_adapter_tf_50
```

Use vLLM only for fast base prompt optimization and future self-distillation
trace generation:

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --k 7 \
  --max-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-thinking \
  --output-path results/qlora_base_promptopt_vllm_eval_50.jsonl \
  --tracker-eval-id base_vllm_promptopt_50
```

The tracker is updated automatically after eval. Use `--no-update-tracker` if
you do not want that, or use `--tracker-eval-id` to overwrite a stable row in
`docs/QLORA_RESULTS_TRACKER.md`.
