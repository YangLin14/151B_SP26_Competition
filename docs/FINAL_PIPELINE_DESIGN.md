# Final Competition Pipeline Design

## Decision

Use the required base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Run pure model inference with vLLM, `enable_thinking=True`, and self-consistency
majority voting. The single entry point is:

```python
from run_inference import run_inference

run_inference()
```

This writes:

```text
results/submission_final.csv
results/submission_final.raw.jsonl
results/submission_final.metadata.json
```

## Why This Pipeline

The competition allows model-intrinsic inference-time strategies such as:

- prompt engineering
- chain-of-thought / thinking mode
- self-consistency
- supervised fine-tuning or RL if trained

The competition does not allow tool-augmented generation at test time. For that
reason, the final pipeline does not use:

- Python execution of model-generated code
- SymPy or calculator calls for private inference
- external APIs
- manual answer editing

Previous Python-first experiments are useful for local analysis, but they are
not the final submission path because they execute generated Python with
`subprocess`.

## Inference Flow

`run_inference()` performs the full end-to-end pipeline:

1. Load `data/private.jsonl`.
2. Build a prompt for each row.
3. Load `Qwen/Qwen3-4B-Thinking-2507` with vLLM.
4. Generate `k` independent responses per question.
5. Process prompts in internal chunks so the full private set does not need to
   be scheduled as one large request batch.
6. Extract the final `\boxed{...}` answer from each generated response.
7. Retry low-confidence questions, then combine original and retry samples:
   - no extractable answer
   - tied vote
   - truncated generation
8. Vote among extracted answers:
   - MCQ: vote on the extracted capital letter.
   - Free-form: vote on normalized boxed content.
9. Select one original full model response whose answer matches the vote.
10. Write Kaggle CSV with exactly:

```text
id,response
```

The submitted `response` field is a full model output trace, not a synthetic
answer-only string.

For DSMLP wall-time limits, `run_inference.py` supports:

- per-chunk raw JSONL checkpoints
- per-checkpoint health summaries printed to stdout
- per-checkpoint public partial accuracy when labels are available
- automatic resume from `--raw-output-path`
- `--start-index` / `--end-index` private-set shards
- `merge_submission_shards.py` for combining partial CSVs

This keeps the final result reproducible while allowing multiple DSMLP sessions
to complete the private set.

## Final A30 Command

The final private run that has been confirmed to start successfully on the A30
uses these high-context self-consistency settings:

```text
model_id = Qwen/Qwen3-4B-Thinking-2507
vllm = 0.9.1
k = 5
max_tokens = 24576
generation_chunk_size = 32
retry_bad = true
retry_k = 2
retry_max_tokens = 32768
temperature = 0.6
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
max_model_len = 32768
gpu_memory_utilization = 0.90
max_num_seqs = 8
max_num_batched_tokens = 16384
enable_chunked_prefill = true
enable_prefix_caching = false
enable_thinking = true
```

`run_inference.py` uses these same settings as its CLI and Python API defaults
so the single entry point reproduces the final submission path. For smaller
debug runs, pass `--limit`, `--start-index`, or `--end-index`.

The environment pins vLLM to `0.9.1` because the older `0.7.x` stack falls back
to the Transformers backend for `Qwen3ForCausalLM`. The script fails fast on
old vLLM versions by default so private inference does not accidentally run on
the slow fallback path.

## Public Parameter Sweep

`sweep_inference_configs.py` automates public validation for these candidate
settings:

```text
A. k=3, max_tokens=24576
B. k=5, max_tokens=24576
C. k=7, max_tokens=24576, generation_chunk_size=16
D. k=5, max_tokens=32768
E. k=5, retry_k=4
```

It calls `run_inference()` for each candidate, reads the generated metadata, and
writes:

```text
results/sweeps/public_inference/summary.csv
results/sweeps/public_inference/summary.json
```

The best config is selected primarily by public overall accuracy, with
boxed-answer coverage, `</think>` completion rate, thinking-token p95,
truncation rate, and retry rate used as secondary diagnostics.

The sweep runs each candidate in a separate Python subprocess rather than
calling all candidates inside one long-lived process. This matters on A30:
vLLM/CUDA memory is released when each subprocess exits, so a heavy candidate is
less likely to poison the next run. Failed candidates are recorded in the summary
with `status=failed` and do not block the rest of the sweep unless
`--no-continue-on-failure` is used.

Chunking does not directly improve single-sample accuracy. It makes the run more
stable on A30 and lets the pipeline add targeted extra samples for questions
where the first pass has no boxed answer, a tie, or a length-truncated output.
That retry step is the accuracy optimization.

The confirmed A30 high-context settings use `max_model_len=32768`,
`max_num_batched_tokens=16384`, and `gpu_memory_utilization=0.90`. Avoid
raising `max_num_batched_tokens` to `32768` on A30 unless you have already
validated it in a short public shard, because vLLM allocates a large KV cache up
front. If the confirmed setup still hits out-of-memory, use the emergency
settings:

```text
k = 1
max_tokens = 4096
max_model_len = 8192
gpu_memory_utilization = 0.60
max_num_seqs = 1
max_num_batched_tokens = 4096
enable_prefix_caching = false
retry_bad = false
```

## Why Not Use The Current QLoRA Adapter

The `yang-test` branch contains useful QLoRA infrastructure, but the documented
results do not justify using an adapter for the final run under time pressure:

```text
Base Qwen, Transformers, thinking 4096, n=50: 26 / 50 = 52%
Numina 5k adapter, same split/settings, n=50: 17 / 50 = 34%
```

Earlier 2048-token adapter comparison showed a small overall gain, but it was
not stable and free-form accuracy was worse. With five days left and A30 access,
the safer final strategy is a strong pure base-model self-consistency pipeline.

## Reproducibility Notes

The pipeline writes metadata next to the CSV, including:

- model id
- data path
- number of questions
- generation hyperparameters
- chunk and retry settings
- elapsed time
- optional public-set score if run on `data/public.jsonl`
- generation health summary:
  - boxed-answer coverage
  - truncation rates
  - retry rate
  - vote-status counts
  - average samples and tokens

vLLM sampling is stochastic, so exact text may differ between runs. The expected
behavior should be consistent because all hyperparameters are fixed in code and
documented here.
