# Gradescope Submission Checklist

## Required Repository Items

- Public GitHub repository link submitted to Gradescope.
- All group members added to the Gradescope submission.
- `README.md` includes:
  - GPU type used.
  - Approximate total generation/inference time.
  - Model weight setup instructions.
  - How to call `run_inference()`.
- `run_inference.py` exists at repo root.
- `run_inference()` produces the final CSV without manual steps.

## Final Entry Point

Python call:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_path="results/submission_final.csv",
)
```

CLI call:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv
```

## Compliance Notes

The final pipeline uses only:

- `Qwen/Qwen3-4B-Thinking-2507`
- vLLM for serving the required model
- prompt engineering
- thinking-mode generation
- self-consistency majority voting over model-generated answers
- format-preserving selection of one original full model response

The final pipeline does not use:

- external model calls
- external APIs
- Python execution of model-generated code
- calculators or SymPy during private inference
- manual correction of answers

## Before Submission

Run:

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
  --no-enable-prefix-caching \
  --generation-chunk-size 64 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 4096
```

Then validate:

```bash
python - <<'PY'
import csv, json
from pathlib import Path
private = {json.loads(line)["id"] for line in Path("data/private.jsonl").open()}
rows = list(csv.DictReader(Path("results/submission_final.csv").open(newline="", encoding="utf-8")))
ids = [int(row["id"]) for row in rows]
assert len(rows) == len(private)
assert set(ids) == private
assert len(ids) == len(set(ids))
assert all(row["response"].strip() for row in rows)
print("submission_final.csv is structurally valid")
PY
```
