# CSE 151B Competition Final Submission

This repository exposes one reproducible inference entry point for the CSE 151B
Spring 2026 Kaggle competition:

```python
from run_inference import run_inference

run_inference()
```

The call loads the required model, runs inference on `data/private.jsonl`,
applies answer extraction, self-consistency voting, adaptive retry, and writes
the final Kaggle CSV to `results/submission_final.csv`. By default, this entry
point uses A30/A100-safe batching parameters. The submitted CSV was generated on
H200 with larger batching parameters listed below.

## Final Method

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Weights: designated base model from HuggingFace Hub; no fine-tuned checkpoint
- Backend: vLLM
- GPU used for final submitted k=5 run: NVIDIA H200
- Approximate generation time: about 174 H200-minutes total, run as two private
  shards of about 87 minutes each; parallel wall-clock time was about 90 minutes
- Test-time tools: none

The final pipeline does not execute model-generated Python code and does not use
calculators, SymPy, external APIs, manual answer correction, or any alternative
model at private inference time.

## Model Weights

No local custom checkpoint is required. vLLM and Transformers download the base
model automatically from HuggingFace Hub:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Use the default HuggingFace cache, or set `HF_HOME` before running:

```bash
export HF_HOME=/path/to/hf_cache
```

## Environment Setup

Use Python 3.11. The setup below is intended for A30/A100-class Linux GPU
machines. Start from a fresh environment if the machine has an older vLLM stack,
especially `vllm==0.7.x`, because old vLLM versions fall back to a slow
Transformers implementation for Qwen3.

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate

uv pip install -r requirements-a30.txt --torch-backend=auto
uv pip check
```

Quick version check:

```bash
python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
assert tuple(map(int, vllm.__version__.split(".")[:3])) >= (0, 9, 1)
PY
```

## Running On A30 / A100

`run_inference.py` defaults to conservative A30/A100 batching parameters. These
defaults keep the same k=5 method, prompts, sampling settings, voting, and retry
logic as the submitted H200 run, but strongly reduce vLLM memory pressure. The
tradeoff is much longer runtime.

The submitted k=5 CSV was generated on H200 with larger batching parameters.
Those exact H200 settings may OOM on A30/A100, especially:

```text
max_num_seqs = 32
max_num_batched_tokens = 32768
generation_chunk_size = 64
max_tokens = 24576
retry_max_tokens = 32768
```

Parameter differences:

| Parameter | A30/A100 default | H200 submitted run |
|---|---:|---:|
| `k` | `5` | `5` |
| `max_tokens` | `24576` | `24576` |
| `retry_bad` | `true` | `true` |
| `retry_k` | `2` | `2` |
| `retry_max_tokens` | `32768` | `32768` |
| `max_model_len` | `32768` | `32768` |
| `gpu_memory_utilization` | `0.90` | `0.90` |
| `max_num_seqs` | `1` | `32` |
| `max_num_batched_tokens` | `4096` | `32768` |
| `generation_chunk_size` | `8` | `64` |
| `enable_prefix_caching` | `false` | `false` |

A30/A100 default command:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv \
  --k 5 \
  --max-tokens 24576 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --generation-chunk-size 8 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 32768
```

If this still OOMs, reduce in this order:

- `--max-num-batched-tokens 2048`
- `--gpu-memory-utilization 0.85`
- as a last resort, lower `--max-tokens` and `--retry-max-tokens`

The full private run can also be split by index range:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_part_000_472.csv \
  --raw-output-path results/submission_part_000_472.raw.jsonl \
  --metadata-path results/submission_part_000_472.metadata.json \
  --start-index 0 \
  --end-index 472 \
  --k 5 \
  --max-tokens 24576 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --generation-chunk-size 8 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 32768
```

Rerunning the same command with the same `--raw-output-path` resumes from saved
raw JSONL checkpoints.

## H200 Submitted Settings

Python API:

```python
from run_inference import run_inference

run_inference(
    max_num_seqs=32,
    max_num_batched_tokens=32768,
    generation_chunk_size=64,
)
```

The no-argument `run_inference()` call uses the A30/A100 defaults. To reproduce
the submitted H200 batching settings, pass the larger H200 parameters explicitly
as above, or use the CLI below.

Equivalent explicit H200 CLI:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv \
  --k 5 \
  --max-tokens 24576 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --no-enable-prefix-caching \
  --generation-chunk-size 64 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 32768
```

These H200 batching settings are not the defaults because they can OOM on
A30/A100.

Output files:

```text
results/submission_final.csv
results/submission_final.raw.jsonl
results/submission_final.metadata.json
```

## Submitted CSV Validation

The final submitted k=5 CSV was produced from two shards:

```text
results/submission_part_000_472.csv
results/submission_part_472_943.csv
```

They were merged with:

```bash
python merge_submission_shards.py \
  --private-path data/private.jsonl \
  --pattern "results/submission_part_*.csv" \
  --output-path results/submission_final.csv
```

Validation result:

```text
required ids: 943
csv rows: 943
order matches private: True
missing: 0
extra: 0
duplicates: 0
empty responses: 0
```

## Contents

| File | Description |
|---|---|
| `run_inference.py` | Final single-entry k=5 inference pipeline |
| `merge_submission_shards.py` | Merge and validate partial private CSV shards |
| `sweep_inference_configs.py` | Public validation sweep utility used to choose settings |
| `docs/FINAL_PIPELINE_DESIGN.md` | Method design, compliance notes, and hyperparameters |
| `docs/GRADESCOPE_SUBMISSION_CHECKLIST.md` | Final submission checklist |
| `judger.py` | Response scoring logic for labeled public data |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `data/private.jsonl` | Private dataset for Kaggle submission |
