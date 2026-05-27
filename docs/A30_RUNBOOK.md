# A30 Runbook

This is the complete command sequence for running the final legal pipeline on
one NVIDIA A30 GPU.

## 1. Confirm Branch And Files

From the repository root:

```bash
git status --short --branch
```

Expected branch:

```text
final-a30-run-inference
```

Confirm the private data exists:

```bash
ls -lh data/private.jsonl
python - <<'PY'
from pathlib import Path
print(sum(1 for _ in Path("data/private.jsonl").open()))
PY
```

Expected count in this repo:

```text
943
```

## 2. Create Environment

Use Python 3.11 on the A30 Linux machine:

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
```

Install CUDA PyTorch:

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install inference and scoring dependencies:

```bash
uv pip install sympy numpy transformers vllm tqdm bitsandbytes \
  antlr4-python3-runtime==4.11.1 ipykernel jupyter accelerate \
  peft trl datasets sentencepiece protobuf scipy pandas scikit-learn \
  -c constraints.txt
```

## 3. Verify GPU And Imports

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
python - <<'PY'
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
print("vLLM and Transformers import OK")
PY
```

Expected GPU type:

```text
NVIDIA A30
```

## 4. Optional Public Smoke Test

Run 10 public examples first. This checks model loading, generation, CSV
writing, raw JSONL writing, and public scoring.

```bash
python run_inference.py \
  --data-path data/public.jsonl \
  --output-path results/public_smoke_submission.csv \
  --limit 10 \
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

Inspect outputs:

```bash
ls -lh results/public_smoke_submission.csv \
       results/public_smoke_submission.raw.jsonl \
       results/public_smoke_submission.metadata.json

python - <<'PY'
import csv, json
from pathlib import Path
csv_path = Path("results/public_smoke_submission.csv")
rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
print("rows:", len(rows))
print("first id:", rows[0]["id"])
print("has boxed:", "\\boxed{" in rows[0]["response"])
print(json.loads(Path("results/public_smoke_submission.metadata.json").read_text())["score_summary"])
PY
```

## 5. Full Public Validation Run

This is optional but recommended before private submission if time allows:

```bash
python run_inference.py \
  --data-path data/public.jsonl \
  --output-path results/public_full_submission.csv \
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

Read the public score:

```bash
python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("results/public_full_submission.metadata.json").read_text())
print(json.dumps(meta["score_summary"], indent=2))
print("elapsed minutes:", meta["elapsed_seconds"] / 60)
PY
```

The run also prints and stores `generation_summary` in the metadata. Use it to
detect problems:

```text
Boxed coverage any sample   should be high; low values mean final answers are not extractable.
Truncated any sample        high values mean max_tokens is too small.
Adaptive retry used         high values mean many first-pass votes were low confidence.
Vote statuses               majority is best; all_none and tie_first are warning signs.
Average tokens/sample       helps estimate whether outputs are hitting the token budget.
```

To inspect it after any run:

```bash
python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("results/submission_final.metadata.json").read_text())
print(json.dumps(meta["generation_summary"], indent=2))
print(json.dumps(meta.get("score_summary"), indent=2))
PY
```

## 5.1 Automatic Public Parameter Sweep

To run all five planned settings and automatically summarize the best public
validation result:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_inference
```

The sweep runs:

```text
A. k=3, max_tokens=4096
B. k=5, max_tokens=4096
C. k=7, max_tokens=4096, generation_chunk_size=16
D. k=5, max_tokens=6144
E. k=5, retry_k=4
```

Outputs:

```text
results/sweeps/public_inference/summary.csv
results/sweeps/public_inference/summary.json
results/sweeps/public_inference/*.metadata.json
results/sweeps/public_inference/*.csv
results/sweeps/public_inference/*.raw.jsonl
```

The script prints the best config and a private-run command shape using the
winning parameters.

Each sweep candidate runs in its own Python subprocess. This is intentional:
when one vLLM run exits, CUDA memory and NCCL state are released by the process
exit before the next candidate starts. If a candidate OOMs or exits non-zero,
the sweep records it as `status=failed` in that candidate's metadata and keeps
running the remaining candidates by default.

For a faster smoke sweep before the full public run:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_smoke_50 \
  --limit 50
```

To resume after an interrupted sweep, rerun the same command. Existing metadata
files are skipped by default. To force rerun everything:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_inference \
  --no-skip-existing
```

To stop immediately on the first failed candidate:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_inference \
  --no-continue-on-failure
```

## 6. Final Private Run

Before a private run, make sure no stale process is still holding GPU memory:

```bash
nvidia-smi
```

If `nvidia-smi` shows a leftover Python process from a failed run that belongs
to you, terminate it:

```bash
kill -TERM <PID>
sleep 5
nvidia-smi
```

If it is still present after `TERM`, use:

```bash
kill -9 <PID>
```

Run the full private set with the final A30 command. This exact setting has been
confirmed to start successfully on the A30:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv \
  --k 5 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.72 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --generation-chunk-size 32 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 4096
```

This is still one reproducible `run_inference()` pipeline. The chunking is
internal: it produces one final CSV, while avoiding one giant vLLM request batch.
The retry pass only adds samples for low-confidence questions.

If the A30 still runs out of memory, use the emergency single-sample fallback:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_final.csv \
  --k 1 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --generation-chunk-size 32 \
  --no-retry-bad
```

## 6.1 DSMLP 8-Hour Limit: Checkpoint And Split Runs

DSMLP pods can terminate after the wall-time deadline. `run_inference.py` now
checkpoints after every generation chunk by writing the raw JSONL file. If a run
is killed, rerun the exact same command with the same `--raw-output-path`; the
script will resume completed questions by default.

After every checkpoint, the script prints compact health metrics:

```text
boxed_any
truncated_any
retried
vote_statuses
```

If the input data includes answers, such as `data/public.jsonl`, it also prints
partial public accuracy for the completed checkpoint rows:

```text
overall
mcq
free
```

So for public checkpoint testing you do not need a separate Python snippet to
inspect accuracy or generation health.

For long private inference, split the private set into 8 shards. Each shard
produces a partial CSV and raw checkpoint. If a shard is interrupted, rerun the
same command for that shard.

Shard boundaries for the current 943-question private set:

```text
part 1:   0-118
part 2: 118-236
part 3: 236-354
part 4: 354-472
part 5: 472-590
part 6: 590-708
part 7: 708-826
part 8: 826-943
```

Template:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-path results/submission_part_START_END.csv \
  --raw-output-path results/submission_part_START_END.raw.jsonl \
  --metadata-path results/submission_part_START_END.metadata.json \
  --start-index START \
  --end-index END \
  --k 5 \
  --max-tokens 4096 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.72 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --generation-chunk-size 32 \
  --retry-bad \
  --retry-k 2 \
  --retry-max-tokens 4096
```

Concrete commands:

```bash
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_000_118.csv --raw-output-path results/submission_part_000_118.raw.jsonl --metadata-path results/submission_part_000_118.metadata.json --start-index 0 --end-index 118 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_118_236.csv --raw-output-path results/submission_part_118_236.raw.jsonl --metadata-path results/submission_part_118_236.metadata.json --start-index 118 --end-index 236 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_236_354.csv --raw-output-path results/submission_part_236_354.raw.jsonl --metadata-path results/submission_part_236_354.metadata.json --start-index 236 --end-index 354 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_354_472.csv --raw-output-path results/submission_part_354_472.raw.jsonl --metadata-path results/submission_part_354_472.metadata.json --start-index 354 --end-index 472 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_472_590.csv --raw-output-path results/submission_part_472_590.raw.jsonl --metadata-path results/submission_part_472_590.metadata.json --start-index 472 --end-index 590 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_590_708.csv --raw-output-path results/submission_part_590_708.raw.jsonl --metadata-path results/submission_part_590_708.metadata.json --start-index 590 --end-index 708 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_708_826.csv --raw-output-path results/submission_part_708_826.raw.jsonl --metadata-path results/submission_part_708_826.metadata.json --start-index 708 --end-index 826 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
python run_inference.py --data-path data/private.jsonl --output-path results/submission_part_826_943.csv --raw-output-path results/submission_part_826_943.raw.jsonl --metadata-path results/submission_part_826_943.metadata.json --start-index 826 --end-index 943 --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
```

After all shards finish, merge them:

```bash
python merge_submission_shards.py \
  --private-path data/private.jsonl \
  --pattern "results/submission_part_*.csv" \
  --output-path results/submission_final.csv
```

## 7. Validate Final CSV

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

private_ids = [json.loads(line)["id"] for line in Path("data/private.jsonl").open(encoding="utf-8")]
with Path("results/submission_final.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row_ids = [int(row["id"]) for row in rows]
missing = sorted(set(private_ids) - set(row_ids))
extra = sorted(set(row_ids) - set(private_ids))
duplicates = sorted({qid for qid in row_ids if row_ids.count(qid) > 1})
empty = [row["id"] for row in rows if not row["response"].strip()]
no_boxed = [row["id"] for row in rows if "\\boxed{" not in row["response"]]

print("required ids:", len(private_ids))
print("csv rows:", len(rows))
print("missing:", missing[:20], "count", len(missing))
print("extra:", extra[:20], "count", len(extra))
print("duplicates:", duplicates[:20], "count", len(duplicates))
print("empty responses:", empty[:20], "count", len(empty))
print("responses without boxed:", no_boxed[:20], "count", len(no_boxed))

if missing or extra or duplicates or empty:
    raise SystemExit("CSV validation failed")
PY
```

`responses without boxed` should ideally be near zero. It is reported but does
not abort because the evaluator may still extract answers from explicit prose in
some cases. If it is high and memory allows it, rerun with larger `--max-tokens`,
such as `6144`.

## 8. Gradescope / README Details

Record these values in the final README:

```text
GPU type: NVIDIA A30
Model: Qwen/Qwen3-4B-Thinking-2507
Inference backend: vLLM
Final command: python run_inference.py --data-path data/private.jsonl --output-path results/submission_final.csv --k 5 --max-tokens 4096 --max-model-len 8192 --gpu-memory-utilization 0.72 --max-num-seqs 4 --max-num-batched-tokens 8192 --no-enable-prefix-caching --generation-chunk-size 32 --retry-bad --retry-k 2 --retry-max-tokens 4096
Approx total generation time: copy from results/submission_final.metadata.json elapsed_seconds
Model weights: downloaded from HuggingFace Hub automatically by Transformers/vLLM cache
Single entry point: run_inference.run_inference()
```
