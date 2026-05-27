# QLoRA Training And Results Tracker

Use this file to track meaningful QLoRA training runs and eval results without
committing large `outputs/` or `results/` artifacts.

## Decision Rule

Only compare runs that use the same eval split, backend, prompt mode, token
budget, and sampling settings.

Current fair comparison target:

```text
Base Transformers eval
vs
Adapter Transformers eval
on outputs/qlora_sft_public_smoke/public_dev_split.jsonl
```

Do not compare base vLLM numbers directly against adapter Transformers numbers.
vLLM is useful for fast base prompt testing and pseudo-label generation, but it
currently cannot run this Qwen3 LoRA adapter in our setup.

## Training Runs

| Run name | Status | Data source | Train examples | Max steps | Seq len | LR | LoRA r/alpha/dropout | Effective batch | Command / notes |
|---|---:|---|---:|---:|---:|---:|---|---:|---|
| `qlora_sft_public_smoke` | done | public | 200 | 50 | 1024 or 2048 | 2e-4 | 16 / 32 / 0.05 | 8 | Smoke test only. Not a competition candidate. |
| `qlora_sft_numina_5k_2048` | done | NuminaMath-CoT | 5000 | -1 | 2048 | 2e-4 | 16 / 32 / 0.05 | 8 | First real adapter. Output: `outputs/qlora_sft_numina_5k_2048/final_adapter`. |
| `qlora_sft_numina_20k_4096_lr1e-4_r32` | suggested | NuminaMath-CoT | 20000 | -1 | 4096 | 1e-4 | 32 / 64 / 0.05 | 8 | Larger reasoning adapter. Use only if 5k is close to base or shows useful behavior. |
| `qlora_sft_numina_50k_4096_lr1e-4_r32` | optional | NuminaMath-CoT | 50000 | -1 | 4096 | 1e-4 | 32 / 64 / 0.05 | 8 | More expensive scale-up. Run after 20k proves useful. |
| `qlora_sft_selfdistill_public_v1` | planned | public correct traces | TBD | -1 | 4096 | 1e-4 | 32 / 64 / 0.05 | 8 | Needs generated correct reasoning traces first. Most likely path if base prompt beats Numina adapter. |

## Eval Results

Fill in `avg tokens`, `boxed`, and `truncated` from the eval script summary.
New eval/scoring scripts update this table automatically after they save results.
Use `--no-update-tracker` to disable that behavior, or `--tracker-eval-id` to
force a stable row id.

| Eval id | Train run / model | Backend | Adapter? | Eval split | n | Prompt mode | Token budget | k | Temp / top_p / top_k | MCQ | Free-form | Overall | Avg tokens | Boxed | Truncated | Output path | Notes |
|---|---|---|---|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `numina_5k_adapter_tf_50` | `qlora_sft_numina_5k_2048` | Transformers | yes | `outputs/qlora_sft_public_smoke/public_dev_split.jsonl` | 50 | current / thinking | max input 2048, max new 2048 | 1 | 0.6 / 0.95 / 20 | 6 / 18 (33.33%) | 14 / 32 (43.75%) | 20 / 50 (40.00%) | TBD | TBD | TBD | TBD | Adapter result from current run. |
| `base_tf_same_as_numina_5k_50` | base `Qwen/Qwen3-4B-Thinking-2507` | Transformers | no | same as above | 50 | same as adapter | same as adapter | 1 | same as adapter | 2 / 18 (11.11%) | 15 / 32 (46.88%) | 17 / 50 (34.00%) | TBD | TBD | TBD | TBD | Fair control for `qlora_sft_numina_5k_2048`; adapter is +3 overall but worse on free-form. |
| `numina_5k_adapter_tf_4096_50` | `qlora_sft_numina_5k_2048` | Transformers | yes | `outputs/qlora_sft_public_smoke/public_dev_split.jsonl` | 50 | thinking | max input 2048, max new 4096 | 1 | 0.6 / 0.95 / 20 | 6 / 18 (33.33%) | 11 / 32 (34.38%) | 17 / 50 (34.00%) | 494.66 | 49 / 50 (98.00%) | any 0 / 50 (0.00%), all 0 / 50 (0.00%) | `results/qlora_numina_5k_2048_thinking4096_eval_50.jsonl` | Adapter 4096 eval; rejected because base 4096 is much stronger. |
| `base_tf_thinking4096_50` | base `Qwen/Qwen3-4B-Thinking-2507` | Transformers | no | `outputs/qlora_sft_public_smoke/public_dev_split.jsonl` | 50 | thinking | max input 2048, max new 4096 | 1 | 0.6 / 0.95 / 20 | 6 / 18 (33.33%) | 20 / 32 (62.50%) | 26 / 50 (52.00%) | 2802.80 | 29 / 50 (58.00%) | any 0 / 50 (0.00%), all 0 / 50 (0.00%) | `results/qlora_base_thinking4096_transformers_eval_50.jsonl` | Current strongest fair baseline. |
| `base_vllm_promptopt_50` | base `Qwen/Qwen3-4B-Thinking-2507` | vLLM | no | same as above | 50 | prompt optimized / thinking | max tokens 32768 | 7 | 0.6 / 0.95 / 20 | TBD | TBD | TBD | TBD | TBD | TBD | `results/qlora_base_promptopt_vllm_eval_50.jsonl` | Not comparable to adapter. Use for prompt optimization and possible self-distillation. |

## Recommended Meaningful Runs

### 1. Finish Fair Base Control For Current Adapter

This answers whether `qlora_sft_numina_5k_2048` is actually better than base
under the same backend/settings.

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

### 2. Re-evaluate Current Adapter With More Output Budget

Use this if the summary shows high truncation or low boxed coverage.

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_numina_5k_2048/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 4096 \
  --enable-thinking \
  --output-path results/qlora_numina_5k_2048_thinking4096_eval_50.jsonl \
  --tracker-eval-id numina_5k_adapter_tf_4096_50
```

Fair base control for that exact setting:

```bash
python scripts/qlora_transformers_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 4096 \
  --enable-thinking \
  --output-path results/qlora_base_thinking4096_transformers_eval_50.jsonl \
  --tracker-eval-id base_tf_thinking4096_50
```

### 3. Larger Numina Reasoning Adapter

Run this only if the 5k adapter is competitive with base or its error profile
looks useful.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_numina_20k_4096_lr1e-4_r32 \
  --data-source numina \
  --max-train-examples 20000 \
  --max-steps -1 \
  --max-seq-len 4096 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --numina-shuffle-buffer 50000
```

Eval:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_numina_20k_4096_lr1e-4_r32/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 4096 \
  --enable-thinking \
  --output-path results/qlora_numina_20k_4096_lr1e-4_r32_eval_50.jsonl \
  --tracker-eval-id numina_20k_adapter_tf_4096_50
```

### 4. Base vLLM Prompt-Optimized Run

This is not a fair adapter comparison. Use it to find a strong base prompt and
to generate candidate reasoning traces for future self-distillation.

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

### 5. Planned Self-Distillation Direction

If base prompt optimization beats the adapter, the next training dataset should
come from correct base generations:

```text
1. Run base with the best prompt on the public train split.
2. Score generated answers.
3. Keep only correct generations with good boxed formatting.
4. Train QLoRA on those correct reasoning traces.
5. Evaluate against the same held-out split with Transformers.
```

This is more promising than scaling answer-only public SFT, because answer-only
training teaches short outputs rather than reliable reasoning.

## How To Get Public Correct Traces

Public correct traces means:

```text
question + base model generated reasoning + final boxed answer
where the generated answer is judged correct against public labels
```

The exact flow is:

1. Use the public train split as the input data, not the held-out dev split.
   If it does not exist yet, create it by running any public QLoRA smoke train.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_smoke \
  --data-source public \
  --max-train-examples 200 \
  --max-steps 50 \
  --max-seq-len 2048
```

This writes:

```text
outputs/qlora_sft_public_smoke/public_train_split.jsonl
outputs/qlora_sft_public_smoke/public_dev_split.jsonl
```

2. Run the strongest base inference setup on the public train split.

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_train_split.jsonl \
  --n-eval -1 \
  --k 7 \
  --max-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-thinking \
  --output-path results/public_train_base_promptopt_vllm.jsonl \
  --tracker-eval-id public_train_base_promptopt_vllm
```

This produces both:

```text
results/public_train_base_promptopt_vllm.raw.jsonl
results/public_train_base_promptopt_vllm.jsonl
```

3. Filter to correct rows only.

For each row in `results/public_train_base_promptopt_vllm.jsonl`, keep rows
where:

```text
correct == true
voted is not null
rep_response contains useful reasoning
finish_reasons are not all "length"
```

4. Convert those rows into SFT training conversations:

```text
system: same training/eval math-solver system prompt
user: original question with options if present
assistant: rep_response from the correct base generation
```

5. Train QLoRA on that filtered trace dataset.

Build the filtered trace dataset:

```bash
python scripts/qlora_build_trace_dataset.py \
  --data-path outputs/qlora_sft_public_smoke/public_train_split.jsonl \
  --results-path results/public_train_base_promptopt_vllm.jsonl \
  --output-path data/public_correct_traces.jsonl
```

Train on the filtered traces:

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_selfdistill_public_v1 \
  --data-source traces \
  --trace-data-path data/public_correct_traces.jsonl \
  --max-train-examples -1 \
  --max-steps -1 \
  --max-seq-len 4096 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --lora-alpha 64
```

Evaluate:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_selfdistill_public_v1/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 4096 \
  --enable-thinking \
  --output-path results/qlora_selfdistill_public_v1_eval_50.jsonl \
  --tracker-eval-id selfdistill_public_v1_tf_4096_50
```
