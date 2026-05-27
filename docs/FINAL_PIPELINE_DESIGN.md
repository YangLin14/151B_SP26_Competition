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

The previous Python-first notebook and `run_colab.py` are useful for local
experimentation, but they are not the final submission path because they execute
generated Python with `subprocess`.

## Inference Flow

`run_inference()` performs the full end-to-end pipeline:

1. Load `data/private.jsonl`.
2. Build a prompt for each row.
3. Load `Qwen/Qwen3-4B-Thinking-2507` with vLLM.
4. Generate `k` independent responses per question.
5. Extract the final `\boxed{...}` answer from each generated response.
6. Vote among extracted answers:
   - MCQ: vote on the extracted capital letter.
   - Free-form: vote on normalized boxed content.
7. Select one original full model response whose answer matches the vote.
8. Write Kaggle CSV with exactly:

```text
id,response
```

The submitted `response` field is a full model output trace, not a synthetic
answer-only string.

## Final A30 Defaults

The default settings in `run_inference.py` are the recommended final A30 setup:

```text
model_id = Qwen/Qwen3-4B-Thinking-2507
k = 5
max_tokens = 8192
temperature = 0.6
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
max_model_len = 32768
gpu_memory_utilization = 0.90
max_num_seqs = 8
max_num_batched_tokens = 32768
enable_chunked_prefill = true
enable_thinking = true
```

If the A30 run hits out-of-memory, use the fallback settings:

```text
k = 3
max_tokens = 8192
gpu_memory_utilization = 0.85
max_num_seqs = 4
max_num_batched_tokens = 16384
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
- elapsed time
- optional public-set score if run on `data/public.jsonl`

vLLM sampling is stochastic, so exact text may differ between runs. The expected
behavior should be consistent because all hyperparameters are fixed in code and
documented here.

