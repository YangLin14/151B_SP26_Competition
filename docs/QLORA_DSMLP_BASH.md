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

Base control:

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-tokens 8192 \
  --output-path results/qlora_base_control_eval_50.jsonl
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
  --output-path results/qlora_base_control_eval_50.jsonl
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
  --output-path results/qlora_base_control_transformers_eval_50.jsonl
```

Public QLoRA adapter:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_public_smoke/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --output-path results/qlora_sft_public_smoke_eval_50.jsonl
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
  --output-path results/qlora_public_v1_base_control_eval_50.jsonl
```

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_public_v1/final_adapter \
  --data-path outputs/qlora_sft_public_v1/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --output-path results/qlora_public_v1_adapter_eval_50.jsonl
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
  --run-name qlora_sft_numina_5k \
  --data-source numina \
  --max-train-examples 5000 \
  --max-steps -1 \
  --max-seq-len 4096 \
  --numina-shuffle-buffer 10000
```

Evaluate larger adapter:

```bash
python scripts/qlora_transformers_eval.py \
  --adapter-path outputs/qlora_sft_numina_5k/final_adapter \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --max-input-length 2048 \
  --max-new-tokens 2048 \
  --output-path results/qlora_numina_5k_eval_50.jsonl
```

## Read Results

```bash
ls -lah results
head -n 2 results/qlora_sft_public_smoke_eval_50.jsonl
cat results/qlora_sft_public_smoke_eval_50.metadata.json
```

Decision rule:

```text
Keep scaling QLoRA only if adapter eval > base control eval on the same held-out examples.
Do not use train-split accuracy as evidence.
Do not compare adapter Transformers numbers directly against base vLLM numbers unless settings match closely.
```
