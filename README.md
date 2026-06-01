# CSE 151B Competition Final Submission

This repository exposes one reproducible inference entry point for the CSE 151B
Spring 2026 Kaggle competition:

```python
from run_inference import run_inference

run_inference()
```

The call loads the required model, runs inference on `data/private.jsonl`,
applies answer extraction, self-consistency voting, adaptive retry, and writes
the final Kaggle CSV to `results/submission_final.csv`.

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

Use Python 3.11. Start from a fresh environment if the machine has an older
vLLM stack, especially `vllm==0.7.x`, because old vLLM versions fall back to a
slow Transformers implementation for Qwen3.

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

## Reproduce The Final Submission

Python API:

```python
from run_inference import run_inference

run_inference()
```

Equivalent explicit CLI:

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

These are also the defaults in `run_inference.py`, so `python run_inference.py`
uses the final submitted configuration.

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
