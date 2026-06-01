# Slide 1 - Title

## Slide text

**A Practical Math Reasoning Submission Pipeline**

Qwen3-4B-Thinking private submission: final H200 K=5 run

Team: Fong-Yu Lin and Anthony Nguyen

## Speaker notes

Our final submission uses the H200 K=5 second-round run split across two shards. The shards merged cleanly into a complete 943-row CSV with no missing, duplicate, extra, or empty responses.

---

# Slide 2 - Summary

## Slide text

- Base model: `Qwen/Qwen3-4B-Thinking-2507`
- Hardware: NVIDIA H200
- Final submitted run:
  - private dataset: 943 questions
  - pure model inference
  - thinking mode enabled
  - `K=5`
  - `max_tokens=24576`
  - adaptive retry
  - two private shards: `0-472` and `472-943`

## Speaker notes

This slide tells the main story. The final submission is the stronger K=5 configuration on H200. It uses the required model, prompt formatting, thinking mode, vLLM inference, self-consistency voting, and adaptive retry. It does not use external tools or Python execution.

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

# Slide 4 - Earlier A30 Backup Run

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

This was the first run we could rely on during development. It is not the final submitted result; the final submission uses the stronger H200 K=5 run.

---

# Slide 5 - Final H200 K=5 Run

## Slide text

Final H200 private split run:

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

This is the submitted run. The H200 has enough headroom for higher concurrency and batch-token capacity while keeping the same legal pure-inference method. The two shards were merged and structurally validated before submission.

---

# Slide 6 - Why K=5 Final?

## Slide text

We used K=5 for the final submission because the public sweep favored it.

- Public 0-150 sweep: 103/150 overall
- MCQ: 47/56
- Free-form: 56/94
- Self-consistency reduces single-sample failures
- Adaptive retry adds samples for low-confidence questions

Tradeoff:

- More compute time and memory than K=1
- Requires shard-level checkpointing and validation

## Speaker notes

K=5 was selected because it gave the strongest completed public validation result while still finishing the private run. The merged final CSV passed structural validation.

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

Final merge diagnostics:

- total submitted rows: 943
- missing ids: 0
- duplicate ids: 0
- empty responses: 0

## Speaker notes

Because private labels are unavailable, we cannot report private accuracy. Instead, our correctness checks are structural, and the final K=5 merge passed them.

---

# Slide 12 - Public Validation Result

## Slide text

- Selected setting: `K=5`, `max_tokens=24576`
- Public 0-150 result: 103/150 overall
- MCQ: 47/56
- Free-form: 56/94
- Boxed coverage any sample: 147/150
- Thinking end any sample: 148/150

## Speaker notes

The public sweep gave us the evidence to use the K=5 configuration for the final submission. It had the best completed public score and acceptable formatting diagnostics.

---

# Slide 13 - Final Pipeline Design

## Slide text

The final pipeline in `run_inference.py` uses:

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

This is the submitted version. Multiple samples and retries reduce single-sample failure, at the cost of more compute time and memory.

---

# Slide 14 - What We Tried But Did Not Submit

## Slide text

Exploratory directions:

- Program-of-Thought with generated Python
- QLoRA / adapter experiments from `yang-test`
- higher-`K` self-consistency beyond K=5
- larger token budgets
- QLoRA fine-tuning

Reason not submitted:

- deadline risk
- incomplete validation
- additional failure modes
- private inference compliance constraints

## Speaker notes

The final submission is not the only idea we considered. We chose the best completed, validated run rather than an unfinished optimized run.

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
- deadline risk was higher than the pure base-model H200 K=5 path

## Speaker notes

The QLoRA work was useful, but it did not justify replacing the base model. The strongest fair comparison was the 4096-token setting, where the base model reached 26/50 and the Numina 5k adapter reached only 17/50.

---

# Slide 16 - Strengths

## Slide text

1. Reproducible private inference path through `run_inference.py`.
2. Legally simple: no tools, no external APIs, no manual correction.
3. Shard-level checkpointing reduces risk from long GPU sessions.
4. Public sweep selected the final K=5 configuration.

## Speaker notes

The main strength is operational reliability. The system is designed around real constraints: GPU memory, wall-clock limits, deadline pressure, and private-set submission format.

---

# Slide 17 - Weaknesses

## Slide text

1. Private accuracy cannot be measured directly.
2. Some generations may miss final boxed answers.
3. QLoRA was attempted but rejected after matched evaluations.
4. K=5 costs substantially more compute than K=1.
5. Shards must be merged and validated carefully.

## Speaker notes

The current submission is stronger than the earlier K=1 development run, but private accuracy is still unknown. The main remaining risks are stochastic variation and imperfect final-answer formatting.

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

- submit the validated H200 K=5 CSV
- keep the exact k=5 `run_inference.py` defaults for reproducibility
- record hardware and runtime in README

Longer term:

- train on correct reasoning traces
- revisit QLoRA only with self-distilled correct traces
- use an adapter only if it beats the base model under matched settings

## Speaker notes

The short-term plan is complete: submit the validated K=5 CSV and keep the repo reproducible. Longer term, self-distillation from correct traces is likely more promising than training only on final answers.

---

# Slide 20 - Final Takeaway

## Slide text

Our final strategy is deadline-aware:

**Submit the validated H200 K=5 split run.**

This gives us:

- a stronger completed submission
- a reproducible inference path
- a clear record of validation and runtime

## Speaker notes

This is the final message. We chose the strongest completed run that passed structural validation and kept the repository aligned with that configuration.
