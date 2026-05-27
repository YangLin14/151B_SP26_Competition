# QLoRA DSMLP / Linux Bash Pipeline

Use this on DSMLP, WSL, Linux servers, or Colab-style Linux environments.

Important: base-model eval can use `vllm`, but Qwen3 currently falls back to
vLLM's Transformers backend in this environment. That backend does not support
LoRA, so adapter eval must use `scripts/qlora_transformers_eval.py`.

Project root:

```bash
cd ~/private/CSE\ 151B/151B_SP26_Competition
```

## Setup

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate peft trl datasets sentencepiece protobuf scipy pandas scikit-learn -c constraints.txt

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "from trl import SFTTrainer, SFTConfig; print('TRL import OK')"
python -c "from vllm import LLM, SamplingParams; print('vLLM import OK')"
```

If vLLM fails with an error like:

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
```

then the installed `vllm` wheel is not ABI-compatible with the installed
`torch`. Fix by reinstalling the pinned torch stack first, then reinstalling
vLLM against that stack:

```bash
uv pip uninstall vllm -y
uv pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --force-reinstall vllm==0.7.3 -c constraints.txt

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "from vllm import LLM, SamplingParams; print('vLLM import OK')"
```

If that still fails on the current DSMLP GPU/node, skip vLLM for now and use
`scripts/qlora_transformers_eval.py`. vLLM is only needed for fast base-model
prompt optimization and public-trace generation; adapter comparison already uses
Transformers.

On H100 MIG nodes, if Transformers generation crashes with `Floating point
exception (core dumped)`, avoid the bitsandbytes 4-bit generation path and use
full bfloat16 plus eager attention:

```bash
python scripts/qlora_transformers_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_train_split.jsonl \
  --n-eval 5 \
  --max-input-length 2048 \
  --max-new-tokens 512 \
  --enable-thinking \
  --no-load-in-4bit \
  --attn-implementation eager \
  --output-path results/node_smoke_transformers_bf16_5.jsonl \
  --tracker-eval-id node_smoke_transformers_bf16_5
```

If Python 3.11 is unavailable but Python 3.12 is installed:

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```

## Public Smoke Training

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_smoke \
  --data-source public \
  --max-train-examples 200 \
  --max-steps 50 \
  --max-seq-len 2048
```

Check output:

```bash
ls -lah outputs/qlora_sft_public_smoke
ls -lah outputs/qlora_sft_public_smoke/final_adapter
cat outputs/qlora_sft_public_smoke/run_metadata.json
```

## Base Evaluation With vLLM

Eval and scoring scripts update `docs/QLORA_RESULTS_TRACKER.md` automatically.
Use `--tracker-eval-id` to update a stable row, or `--no-update-tracker` to
disable tracker updates for a one-off run.

Base control:

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-tokens 8192 \
  --enable-thinking \
  --output-path results/qlora_base_control_eval_50.jsonl \
  --tracker-eval-id base_vllm_control_50
```

Prompt-optimized base control from the prompt branch:

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

The eval script saves raw generations before scoring:

```text
results/qlora_base_control_eval_50.raw.jsonl
```

If scoring fails after generation, score the raw file without rerunning vLLM:

```bash
python scripts/qlora_score_raw.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --raw-path results/qlora_base_control_eval_50.raw.jsonl \
  --output-path results/qlora_base_control_eval_50.jsonl \
  --tracker-eval-id base_vllm_control_50
```

If you hit `ModuleNotFoundError: No module named 'judger'`, update to the latest
script and rerun the eval. Older versions did not save raw generations before
scoring, so that specific failed run cannot be recovered.

Do not use vLLM for LoRA adapter eval in this setup. This fails with:

```text
AssertionError: TransformersModel does not support LoRA yet.
```

## Adapter Evaluation With Transformers

Base control with the same Transformers backend/settings:

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

Public QLoRA adapter:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_public_smoke/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --enable-thinking \
  --output-path results/qlora_sft_public_smoke_eval_50.jsonl \
  --tracker-eval-id public_smoke_adapter_tf_50
```

## Public Real Training

This trains on the public 80% train split and keeps 20% held out.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_v1 \
  --data-source public \
  --max-train-examples -1 \
  --max-steps -1 \
  --max-seq-len 2048
```

Evaluate:

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_v1/public_dev_split.jsonl \
  --n-eval 50 \
  --max-tokens 8192 \
  --enable-thinking \
  --output-path results/qlora_public_v1_base_control_eval_50.jsonl \
  --tracker-eval-id public_v1_base_vllm_50
```

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_public_v1/final_adapter \
  --data-path outputs/qlora_sft_public_v1/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --enable-thinking \
  --output-path results/qlora_public_v1_adapter_eval_50.jsonl \
  --tracker-eval-id public_v1_adapter_tf_50
```

## NuminaMath CoT Training

The script streams NuminaMath by default and saves the selected subset under the run output directory.

Smoke:

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_numina_smoke \
  --data-source numina \
  --max-train-examples 500 \
  --max-steps 50 \
  --max-seq-len 2048 \
  --numina-shuffle-buffer 10000
```

Evaluate smoke adapter:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_numina_smoke/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --output-path results/qlora_numina_smoke_eval_50.jsonl
```

Larger run:

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_numina_5k_2048 \
  --data-source numina \
  --max-train-examples 5000 \
  --max-steps -1 \
  --max-seq-len 2048 \
  --numina-shuffle-buffer 10000
```

Evaluate larger adapter:

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

Larger suggested reasoning run:

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

Evaluate larger suggested adapter:

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

## Read Results

```bash
ls -lah results
head -n 2 results/qlora_sft_public_smoke_eval_50.jsonl
cat results/qlora_sft_public_smoke_eval_50.metadata.json
sed -n '1,120p' docs/QLORA_RESULTS_TRACKER.md
```

## Public Correct Traces / Self-Distillation

Use this only after a base prompt setting beats the adapter. It runs the base
model on the public train split, scores generations, and creates candidate rows
for a future filtered trace dataset.

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

Then keep rows from `results/public_train_base_promptopt_vllm.jsonl` where
`correct == true`, `voted` is not null, and generations are not fully truncated.

Build the filtered trace dataset:

```bash
python scripts/qlora_build_trace_dataset.py \
  --data-path outputs/qlora_sft_public_smoke/public_train_split.jsonl \
  --results-path results/public_train_base_promptopt_vllm.jsonl \
  --output-path data/public_correct_traces.jsonl
```

Train the self-distilled public-trace adapter:

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

Evaluate the self-distilled adapter:

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

Decision rule:

```text
Keep scaling QLoRA only if adapter eval > base control eval on the same held-out examples.
Do not use train-split accuracy as evidence.
Do not compare adapter Transformers numbers directly against base vLLM numbers unless settings match closely.
```
