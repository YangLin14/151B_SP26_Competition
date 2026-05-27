# CSE 151B Competition — Final Submission Pipeline

This repository exposes a single reproducible inference entry point for the
CSE 151B Spring 2026 Kaggle competition.

## Final Method

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Backend: vLLM
- Strategy: pure model inference with thinking mode and self-consistency voting
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

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install sympy numpy transformers vllm tqdm bitsandbytes \
  antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate \
  peft trl datasets sentencepiece protobuf scipy pandas scikit-learn \
  -c constraints.txt
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
  --k 3 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.72 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching
```

Output files:

```text
results/submission_final.csv
results/submission_final.raw.jsonl
results/submission_final.metadata.json
```

## Contents

| File | Description |
|---|---|
| `run_inference.py` | Final single-entry inference pipeline |
| `docs/A30_RUNBOOK.md` | Complete A30 setup, run, and validation commands |
| `docs/FINAL_PIPELINE_DESIGN.md` | Method design, compliance notes, and hyperparameters |
| `docs/GRADESCOPE_SUBMISSION_CHECKLIST.md` | Final submission checklist |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `data/private.jsonl` | Private dataset for Kaggle submission |
| `results/` | Runtime output directory created by `run_inference.py` |
