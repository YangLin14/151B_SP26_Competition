# Slide 1 - Title

## Slide text

**A Practical Math Reasoning Submission Pipeline**

Qwen3-4B-Thinking private submission: A30 backup run and H200 K=5 second round

Team: Fong-Yu Lin and Anthony Nguyen

## Speaker notes

Our final submission strategy is pragmatic. We first secured a working A30 K=1 private run, then ran a stronger H200 K=5 second round split across two shards. If the H200 shards merge and validate cleanly, that should be the preferred submission; the A30 K=1 result remains the safer backup.

---

# Slide 2 - Summary

## Slide text

- Base model: `Qwen/Qwen3-4B-Thinking-2507`
- Hardware:
  - initial backup: NVIDIA A30
  - stronger second round: H200
- Current submission candidates:
  - backup: A30 K=1 notebook run
  - preferred if validated: H200 K=5 split run
- H200 second round:
  - private dataset: 943 questions
  - pure model inference
  - thinking mode enabled
  - `K=5`
  - `max_tokens=24576`
  - adaptive retry
  - two private shards: `0-472` and `472-943`

## Speaker notes

This slide tells the main story. We chose reliability first with the A30 run, then used the H200 to run the stronger K=5 configuration. Neither path uses external tools or Python execution; both use the required model, prompt formatting, thinking mode, and vLLM inference.

---

# Slide 3 - Problem Setup

## Slide text

- Task: solve math reasoning problems and submit a CSV with:
  - `id`
  - `response`
- Response should contain a full model-generated solution trace.
- Final answer must be extractable from `\boxed{...}`.
- Problem types:
  - multiple choice: one boxed capital letter
  - free form: boxed numeric, symbolic, or ordered multi-answer output

## Speaker notes

The challenge is not just generating a plausible solution. The output must also be machine-readable. We therefore focused heavily on prompt format, boxed-answer extraction, and checkpointed generation.

---

# Slide 4 - Backup A30 Notebook Run

## Slide text

Notebook: `inference_sc_k1_private (1).ipynb`

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Data | `data/private.jsonl` |
| Private rows | 943 |
| GPU | NVIDIA A30 |
| Precision | bfloat16 |
| `K` | 1 |
| `max_tokens` | 16384 |
| `max_model_len` | 32768 |
| `temperature / top_p / top_k` | 0.6 / 0.95 / 20 |

## Speaker notes

This was the first run we could rely on for submission. It is less ambitious than the H200 K=5 run, but it is simpler, cheaper, and configured around the A30 memory budget.

---

# Slide 5 - Stronger H200 Second Round

## Slide text

H200 private split run:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Data | `data/private.jsonl` |
| Shard 1 | `start-index 0`, `end-index 472` |
| Shard 2 | `start-index 472`, `end-index 943` |
| `K` | 5 |
| `max_tokens` | 24576 |
| `max_model_len` | 32768 |
| `max_num_seqs` | 32 |
| `max_num_batched_tokens` | 32768 |
| `generation_chunk_size` | 64 |
| Retry | `retry_bad`, `retry_k=2`, `retry_max_tokens=32768` |
| Prefix caching | disabled |

## Speaker notes

This is the stronger second-round run. The H200 has more headroom than the A30, so we increased concurrency and batch-token capacity while keeping the same legal pure-inference method. The two shards must be merged and structurally validated before submission.

---

# Slide 6 - Why K=1 First?

## Slide text

We prioritized a valid private submission before optimizing accuracy.

- Lower memory pressure on A30
- Faster wall-clock completion
- Fewer out-of-memory risks
- Easier chunk-level recovery
- Avoids waiting for all planned sweeps before submitting

Tradeoff:

- No majority-vote benefit from self-consistency
- More vulnerable to one bad sample per question

## Speaker notes

K=1 was not the best theoretical method. It was the safest first completed run. The plan is to submit the H200 K=5 result only if both shards finish and pass validation; otherwise, use the A30 K=1 result as backup.

---

# Slide 7 - Prompt Design

## Slide text

Two prompt templates:

- **Free-form prompt**
  - one answer per `[ANS]`
  - answers in order
  - all answers inside one `\boxed{}`
  - no units or prose inside the box
- **MCQ prompt**
  - one capital letter inside `\boxed{}`
  - no formula, number, or option text inside the box

## Speaker notes

The prompt is intentionally direct. Instead of asking for broad mathematical explanations only, it emphasizes the final answer contract. That matters because the scorer needs extractable final answers.

---

# Slide 8 - Inference Pipeline

## Slide text

1. Load `data/private.jsonl`.
2. Split 943 questions into 64 chunks.
3. Select assigned chunks or sections.
4. Render Qwen chat-template prompts with `enable_thinking=True`.
5. Generate one response per question with vLLM.
6. Extract the final `\boxed{...}` answer for diagnostics.
7. Save each chunk immediately as a CSV checkpoint.
8. Merge chunk CSVs for final submission.

## Speaker notes

The private set is chunked to reduce deadline risk. If a session dies, we do not lose all progress. Each chunk writes a valid partial CSV, and completed chunks are skipped on rerun.

---

# Slide 9 - Chunking And Checkpointing

## Slide text

- Private set: 943 questions
- Chunking scheme:
  - 8 sections
  - 64 chunks total
  - about 14 questions per chunk
- Output pattern:
  - `results/submission/sc_k1_private_cXX.csv`
- Merge validation:
  - exact row count
  - no missing ids
  - no duplicates
  - no empty responses

## Speaker notes

This design supports splitting work across multiple runners. The merge script validates row count, ids, duplicates, extras, and empty responses.

---

# Slide 10 - Validity And Compliance

## Slide text

The submitted path uses:

- required base model only
- vLLM serving
- prompt engineering
- thinking-mode generation
- chunked inference and checkpointing
- model-generated responses only

It does **not** use:

- external APIs
- external models
- Python execution of generated code
- SymPy or calculators during private inference
- manual answer correction

## Speaker notes

This is important because some exploratory ideas, especially Program-of-Thought, execute generated Python. That is useful for analysis, but it is not part of the private submission path.

---

# Slide 11 - Current Run Diagnostics

## Slide text

Private labels are unavailable, so current validation is structural:

- every private id appears exactly once
- all responses are non-empty
- generated chunks are saved as CSV checkpoints
- final merged CSV passes validation script

Diagnostics to report after final merge:

- total submitted rows
- missing ids
- duplicate ids
- empty responses

## Speaker notes

Because private labels are unavailable, we cannot report private accuracy. Instead, our correctness checks are structural.

---

# Slide 12 - Public Validation Plan

## Slide text

If time allows, run public validation before replacing the submission:

```bash
python sweep_inference_configs.py \
  --data-path data/public.jsonl \
  --output-dir results/sweeps/public_inference
```

Sweep candidates:

- `K=3`, `max_tokens=24576`
- `K=5`, `max_tokens=24576`
- `K=7`, `max_tokens=24576`
- `K=5`, `max_tokens=32768`
- `K=5`, `retry_k=4`

## Speaker notes

The public sweep is the evidence-based way to choose a stronger final configuration. We should only replace the K=1 submission if the stronger run finishes and passes validation in time.

---

# Slide 13 - Stronger Pipeline Design

## Slide text

The stronger pipeline in `run_inference.py` adds:

- self-consistency voting with `K=5`
- longer generation budget: `max_tokens=24576`
- adaptive retry for:
  - no boxed answer
  - tied vote
  - length truncation
- raw JSONL checkpoints
- metadata summaries
- public scoring when labels are available

## Speaker notes

This is the version we prefer if the H200 shards finish and validate. It is more accurate in principle because multiple samples and retries reduce single-sample failure, but it also costs more time and memory.

---

# Slide 14 - What We Tried But Did Not Submit

## Slide text

Exploratory directions:

- Program-of-Thought with generated Python
- QLoRA / adapter experiments from `yang-test`
- higher-`K` self-consistency
- larger token budgets
- public validation sweeps

Reason not submitted yet:

- deadline risk
- incomplete validation
- additional failure modes
- private inference compliance constraints

## Speaker notes

The final submission is not the only idea we considered. We made a conservative final choice because a robust, valid submission is more important than an unfinished optimized run.

---

# Slide 15 - QLoRA Findings From `yang-test`

## Slide text

| Experiment | Fair control | Adapter result | Decision |
|---|---:|---:|---|
| Public smoke adapter, n=10 | Base 4/10 | Adapter 3/10 | Reject |
| Numina 5k, 2048 tokens, n=50 | Base 17/50 | Adapter 20/50 | Not stable |
| Numina 5k, 4096 tokens, n=50 | Base 26/50 | Adapter 17/50 | Reject |

Why QLoRA failed for this submission:

- public answer-only data taught short answers more than reasoning
- Numina adapter did not consistently improve free-form accuracy
- vLLM could not run this LoRA adapter in our setup, so adapter eval required Transformers
- deadline risk was higher than the pure base-model A30 path

## Speaker notes

The QLoRA work was useful, but it did not justify replacing the base model. The strongest fair comparison was the 4096-token setting, where the base model reached 26/50 and the Numina 5k adapter reached only 17/50.

---

# Slide 16 - Strengths

## Slide text

1. Reliable private inference path on a single A30 GPU.
2. Legally simple: no tools, no external APIs, no manual correction.
3. Chunk-level checkpointing reduces risk from long DSMLP sessions.
4. Clear path to stronger runs through public sweeps and adaptive retry.

## Speaker notes

The main strength is operational reliability. The system is designed around real constraints: GPU memory, wall-clock limits, deadline pressure, and private-set submission format.

---

# Slide 17 - Weaknesses

## Slide text

1. `K=1` does not benefit from majority voting.
2. Private accuracy cannot be measured directly.
3. Some generations may miss final boxed answers.
4. QLoRA was attempted but rejected after matched evaluations.
5. Optimized runs may not finish before the deadline.
6. Notebook chunks must be merged and validated carefully.

## Speaker notes

The current submission is conservative, not optimal. The main risk is that one bad generation directly becomes the submitted response.

---

# Slide 18 - Lessons Learned

## Slide text

- A valid submission pipeline matters before model optimization.
- Long-context reasoning needs checkpointing and diagnostics.
- Output format can be as important as reasoning quality.
- Backend details matter: native Qwen3 support in vLLM is critical.
- Public validation should guide which expensive private run to trust.
- Fine-tuning is not automatically better than a strong base reasoning model.

## Speaker notes

This project became as much an engineering problem as a modeling problem. We learned that large-model inference requires careful operational design, not just better prompts.

---

# Slide 19 - Future Work

## Slide text

Immediate:

- merge all K=1 private chunks
- validate final CSV structure
- submit the K=1 notebook result

Next if time allows:

- finish public sweep
- run K=5 or K=7 private inference
- enable adaptive retry
- compare public diagnostics before replacing submission

Longer term:

- train on correct reasoning traces
- revisit QLoRA only with self-distilled correct traces
- use an adapter only if it beats the base model under matched settings

## Speaker notes

The short-term plan is simple: get the guaranteed submission in, then improve only if a stronger run finishes cleanly. Longer term, self-distillation from correct traces is likely more promising than training only on final answers.

---

# Slide 20 - Final Takeaway

## Slide text

Our final strategy is deadline-aware:

**Use the H200 K=5 split run if both shards merge and validate; keep the A30 K=1 run as a backup.**

This gives us:

- a guaranteed backup submission
- a stronger preferred submission candidate
- a reproducible inference path
- a clear plan for stronger future runs

## Speaker notes

This is the final message. We chose a practical strategy that balances accuracy, reliability, and deadline risk.
