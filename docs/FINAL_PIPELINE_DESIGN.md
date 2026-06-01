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

## Final H200 Command

The final private submission uses these high-context self-consistency settings:

```text
model_id = Qwen/Qwen3-4B-Thinking-2507
vllm = 0.9.1
k = 5
max_tokens = 24576
generation_chunk_size = 64
retry_bad = true
retry_k = 2
retry_max_tokens = 32768
temperature = 0.6
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
max_model_len = 32768
gpu_memory_utilization = 0.90
max_num_seqs = 32
max_num_batched_tokens = 32768
enable_chunked_prefill = true
enable_prefix_caching = false
enable_thinking = true
```

`run_inference.py` uses these same settings as its CLI and Python API defaults
so the single entry point reproduces the final submission path. For smaller
debug runs, pass `--limit`, `--start-index`, or `--end-index`.

The submitted private run was split into two shards:

```text
results/submission_part_000_472.csv
results/submission_part_472_943.csv
```

The merged final CSV passed structural validation:

```text
required ids = 943
csv rows = 943
order matches private = true
missing = 0
extra = 0
duplicates = 0
empty responses = 0
```

The first shard metadata recorded 472 questions in 5208.8 seconds, about 86.8
minutes. The second shard was comparable in size. Total generation time was
therefore about 174 H200-minutes, or about 90 minutes wall-clock when both
shards ran in parallel.

The environment pins vLLM to `0.9.1` because the older `0.7.x` stack falls back
to the Transformers backend for `Qwen3ForCausalLM`. The script fails fast on
old vLLM versions by default so private inference does not accidentally run on
the slow fallback path.

## Public Parameter Sweep

`sweep_inference_configs.py` was used to compare public validation settings.
The selected configuration was `k=5`, `max_tokens=24576`, `retry_k=2`:

```text
A. k=5, max_tokens=24576
B. k=7, max_tokens=24576, generation_chunk_size=16
C. k=3, max_tokens=24576
D. k=5, max_tokens=16384
E. k=5, max_tokens=24576, retry_k=4
```

The sweep writes metadata for each candidate and a summary table:

```text
results/sweeps/public_inference/summary.csv
results/sweeps/public_inference/summary.json
```

On the public 0-150 sweep, the selected `k=5`, `max_tokens=24576` configuration
scored 103/150 overall, with 47/56 MCQ and 56/94 free-form. Boxed coverage and
thinking completion diagnostics were also acceptable for the final private run.

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

If reproducing on smaller GPUs such as A30, lower `max_num_seqs`,
`max_num_batched_tokens`, and `generation_chunk_size` may be necessary for
debugging. Those lower-memory settings are not the submitted configuration; the
submitted H200 k=5 settings above are the reproducibility target.

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
