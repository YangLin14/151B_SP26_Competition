# CSE 151B Competition — Final Submission Pipeline

This repository exposes a single reproducible inference entry point for the
CSE 151B Spring 2026 Kaggle competition.

## Final Method

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Backend: vLLM
- Strategy: pure model inference with thinking mode, self-consistency voting,
  internal chunked generation, and adaptive retry for low-confidence outputs
- Test-time tools: none
- Final entry point: `run_inference.run_inference()`

The final pipeline does not execute model-generated Python code and does not use
calculators, SymPy, external APIs, or any alternative model at inference time.

## Hardware Used

- GPU type: NVIDIA A30
- Approximate full private inference time: fill from
  `results/submission_final.metadata.json` after the final run

## Model Weights

No custom fine-tuned checkpoint is required for the final method. The designated
base model is downloaded automatically from HuggingFace Hub by vLLM /
Transformers:

```text
Qwen/Qwen3-4B-Thinking-2507
```

If running on a shared A30 server, keep the default HuggingFace cache or set:

```bash
export HF_HOME=/path/to/hf_cache
```

## Environment Setup

Start from a fresh environment if the machine already has `vllm==0.7.x` or a
manual `torch==2.5.1` install. vLLM 0.7.x falls back to Transformers for
`Qwen3ForCausalLM`, which is much slower and uses memory differently.

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate

uv pip install -r requirements-a30.txt --torch-backend=auto
uv pip check
```

Do not install PyTorch separately before vLLM. The vLLM wheel is compiled
against a specific PyTorch/CUDA stack, so mixing an old Torch wheel with a newer
vLLM wheel can trigger CUDA or custom-kernel crashes.

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

## Run Final Private Inference

Python API:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_path="results/submission_final.csv",
)
```

Command line:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv \
  --k 5 \
  --max-tokens 24576 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --generation-chunk-size 32 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 32768
```

Output files:

```text
results/submission_final.csv
results/submission_final.raw.jsonl
results/submission_final.metadata.json
```

## Run Public Parameter Sweep

To compare the five planned public validation settings and automatically pick
the best one:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_inference
```

For a quick smoke sweep, add `--limit 50`.

## Contents

| File | Description |
|---|---|
| `run_inference.py` | Final single-entry inference pipeline |
| `sweep_inference_configs.py` | Public validation sweep for inference settings |
| `merge_submission_shards.py` | Merge partial private CSV shards after split DSMLP runs |
| `docs/A30_RUNBOOK.md` | Complete A30 setup, run, and validation commands |
| `docs/FINAL_PIPELINE_DESIGN.md` | Method design, compliance notes, and hyperparameters |
| `docs/GRADESCOPE_SUBMISSION_CHECKLIST.md` | Final submission checklist |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `data/private.jsonl` | Private dataset for Kaggle submission |
| `results/` | Runtime output directory created by `run_inference.py` |
