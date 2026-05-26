# Slide 1 - Title

## Slide text

**Improving Mathematical Reasoning for the CSE 151B Competition**

Qwen3-4B-Thinking Inference, Prompt Optimization, Program-of-Thought, and QLoRA

Team: [fill in names]

## Speaker notes

Our goal was to improve mathematical reasoning accuracy for the CSE 151B competition. We approached the problem from four directions: building a reliable baseline, improving prompts and output formatting, using vLLM for faster inference and self-consistency, and training QLoRA adapters through supervised fine-tuning.

---

# Slide 2 - Summary

## Slide text

- We built a reproducible math-reasoning pipeline around `Qwen/Qwen3-4B-Thinking-2507`.
- The public dataset contains 1,126 questions: 375 multiple-choice and 751 free-form.
- Our strongest tested direction was prompt/vLLM optimization, reaching 26/50 on a 50-question public subset in the optimization notebook.
- QLoRA training worked end-to-end, but early adapters only modestly improved the base control and did not consistently improve free-form answers.

## Speaker notes

This slide gives the high-level story. The key point is not just that we used a large language model, but that we made the experiment pipeline reproducible: data splitting, inference, scoring, result tracking, and backend-aware comparisons. In our current results, prompt and vLLM optimization gave the strongest short-term gains, while QLoRA worked end-to-end but still needs better reasoning traces to become consistently useful.

---

# Slide 3 - Team Introduction

## Slide text

- [Name 1]: baseline notebook, dataset inspection, result validation
- [Name 2]: vLLM inference, prompt optimization, self-consistency experiments
- [Name 3]: QLoRA training pipeline, adapter evaluation, result tracker
- [Name 4]: classical baselines, report/presentation, error analysis

## Speaker notes

Replace these placeholders with the real team members and responsibilities. It is better to list each person's technical contribution instead of only listing names, because it makes the collaboration and project coverage clearer.

---

# Slide 4 - Problem And Dataset

## Slide text

- Task: solve mathematical reasoning problems and return a final boxed answer.
- Two formats:
  - Multiple-choice: output one letter, e.g. `\boxed{C}`
  - Free-form: output one or more values, e.g. `\boxed{3, 7}`
- Public data:
  - 1,126 total questions
  - 375 MCQ
  - 751 free-form
- Evaluation uses exact MCQ matching plus symbolic/numeric judging for free-form answers.

## Speaker notes

This dataset is not just a classification task. Multiple-choice answers can be evaluated by exact letter matching, but free-form answers may involve equivalent expressions, floating-point tolerance, or different LaTeX formats. We therefore used the project `Judger` to normalize answers and handle symbolic or numeric equivalence.

---

# Slide 5 - Baselines

## Slide text

| Model / Method | Scope | Result |
|---|---:|---:|
| Majority-label baseline | MCQ dev only, n=75 | 10/75 = 13.33% |
| Bag-of-Words Logistic Regression | MCQ dev only, n=75 | 9/75 = 12.00% |
| Bag-of-Words MLP | MCQ dev only, n=75 | 8/75 = 10.67% |
| Starter Qwen smoke run | n=5 | 3/5 = 60.00% |
| Prompt v2 smoke run | n=10 | 5/10 = 50.00% |

## Speaker notes

Classical baselines intentionally perform poorly because math reasoning cannot be solved well from shallow word counts. The starter and prompt smoke runs are small sanity checks, not final claims. Use this slide to show why a deep sequence model is necessary and why we carefully label small runs as smoke tests.

---

# Slide 6 - Methodology Overview

## Slide text

Pipeline:

1. Load public JSONL data and separate MCQ/free-form tasks.
2. Build task-specific chat prompts.
3. Run Qwen3-4B-Thinking with Transformers or vLLM.
4. Extract the last `\boxed{...}` answer.
5. Score with MCQ exact match or `Judger.auto_judge`.
6. Track experiment settings and accuracy in Markdown/JSONL files.

## Speaker notes

This slide explains the overall workflow. The important point is that we did not manually inspect answers one by one. Each run produces JSONL outputs with raw generations, boxed answers, vote status, token counts, and finish reasons, which makes later comparisons more reliable.

---

# Slide 7 - Data Processing

## Slide text

- JSONL loader reads one problem per line.
- MCQ prompts include labeled answer choices.
- Free-form prompts emphasize final boxed answers and ordered multi-answer output.
- Public SFT experiments use an 80/20 train/dev split to avoid train-on-test reporting.
- Result files store per-question fields: `id`, `gold`, `samples_boxed`, `voted`, `correct`, and token diagnostics.

## Speaker notes

This slide should emphasize evaluation fairness. If we fine-tune on the entire public set and then report public accuracy, the result is contaminated by train-on-test leakage. For public QLoRA experiments, we therefore used an 80/20 train/dev split and compared the base model and adapter on the held-out dev split.

---

# Slide 8 - Deep Learning Model

## Slide text

- Base model: `Qwen/Qwen3-4B-Thinking-2507`
- Inference backends:
  - Transformers: adapter-compatible, used for fair QLoRA comparisons
  - vLLM: faster base-model inference and prompt experiments
- Key generation settings:
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`
  - long token budgets up to 32,768 for reasoning-heavy runs

## Speaker notes

vLLM is useful for fast base-model prompt experiments, but in our current Qwen3 setup, LoRA adapter support through vLLM is limited. For fair adapter comparisons, we used Transformers instead. This is an important experimental limitation: numbers from different backends should not be directly compared unless the settings are matched.

---

# Slide 9 - Engineering Tricks

## Slide text

- Task-specific prompts for MCQ vs. free-form answers.
- Forced thinking mode with `enable_thinking=True`.
- Robust answer extraction using the last `\boxed{...}` occurrence.
- Self-consistency via multiple samples and majority vote.
- vLLM throughput settings:
  - prefix caching
  - chunked prefill
  - high GPU memory utilization
- Auto-resume for long-running evaluation scripts.

## Speaker notes

This slide highlights the engineering work. One important detail is extracting the last boxed answer, because a thinking model may produce intermediate boxed expressions before the final answer. Auto-resume was also useful because even a 50-question run can take tens of minutes, and a full public run takes longer.

---

# Slide 10 - Program-of-Thought Direction

## Slide text

Program-of-Thought pipeline:

1. Ask the model to generate a self-contained Python solution.
2. Execute the code with a timeout.
3. If execution fails, retry once with error feedback.
4. If Python still fails, fall back to normal reasoning.
5. Wrap successful stdout as `\boxed{...}` for the same scorer.

## Speaker notes

This direction came from the `python_vllm` branch. It separates the system into a Python-first path and a reasoning fallback path. The advantage is that arithmetic and symbolic computation can be handled by Python or SymPy. The downside is that code extraction, execution safety, timeouts, and output formatting become new sources of failure.

---

# Slide 11 - QLoRA Training Method

## Slide text

- 4-bit NF4 quantization with bfloat16 compute.
- LoRA attached to attention and MLP projection layers:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`
- Main runs:
  - `qlora_sft_public_smoke`: 200 public examples, 50 steps
  - `qlora_sft_numina_5k_2048`: 5,000 NuminaMath-CoT examples
- Effective batch size: 8 in documented runs.

## Speaker notes

The goal of QLoRA was not to retrain the full 4B model. Instead, we trained low-cost adapters to improve output formatting or reasoning behavior. Public answer-only data has limited reasoning supervision, so we also tested NuminaMath-CoT because it contains worked solutions.

---

# Slide 12 - Experiment 1: Prompt And vLLM Optimization

## Slide text

| Run | Eval size | MCQ | Free-form | Overall |
|---|---:|---:|---:|---:|
| Prompt v2 smoke | 10 | 1/3 | 4/7 | 5/10 = 50.00% |
| Prompt v2 result file | 50 | 6/13 | 1/37 | 7/50 = 14.00% |
| Optimization notebook run | 50 | 8/13 | 18/37 | 26/50 = 52.00% |

## Speaker notes

Be transparent here: different files currently show different run states. `results/prompt_v2_greedy_smoke_50.jsonl` records 7/50, while the notebook output in `origin/model-optimization-v1` shows a 50-question run reaching 26/50. Before the final presentation, the strongest setting should be re-run and saved as a consistent JSONL result.

---

# Slide 13 - Experiment 2: QLoRA Adapters

## Slide text

| Run | Eval split | Backend | Overall |
|---|---|---|---:|
| Base control | public held-out dev, n=10 | Transformers | 4/10 = 40.00% |
| Public QLoRA smoke adapter | same, n=10 | Transformers | 3/10 = 30.00% |
| Base control | public held-out dev, n=50 | Transformers | 17/50 = 34.00% |
| Numina 5k QLoRA adapter | same, n=50 | Transformers | 20/50 = 40.00% |

## Speaker notes

The public smoke adapter proves that the training pipeline works, but it should not be treated as a final model. The Numina 5k adapter improved overall accuracy by 3 questions over the base control, but its free-form accuracy was slightly worse. The correct conclusion is cautious: the adapter shows signal, but it is not yet a consistent win.

---

# Slide 14 - Discussion: Strengths

## Slide text

1. Reproducible experiment pipeline with saved JSONL outputs and result tracker.
2. Strong engineering coverage: vLLM inference, Transformers adapter eval, QLoRA training, and scoring utilities.
3. Careful evaluation design: held-out split for public SFT and backend-aware comparisons.

## Speaker notes

This slide should clearly state what the project did well. We kept experiment records instead of only running notebooks, separated backend-specific comparisons, avoided train-on-test reporting, and added output diagnostics such as boxed-answer coverage and truncation rates.

---

# Slide 15 - Discussion: Weaknesses

## Slide text

1. Some results are still small-sample smoke tests and need full re-runs.
2. QLoRA adapters did not consistently improve free-form mathematical reasoning.
3. Program-of-Thought introduces extra failure modes: bad code blocks, runtime errors, timeouts, and formatting drift.

## Speaker notes

This slide should not hide the limitations. A strong presentation should show that we understand the weaknesses of our own evidence. The most important caveat is that the strongest 26/50 result currently comes from notebook output, so it should be re-run and saved as a JSONL result before the final report.

---

# Slide 16 - Improvements

## Slide text

- Re-run all top candidates on the same 50-question and full public split.
- Convert correct base generations into self-distillation traces.
- Train a QLoRA adapter on correct reasoning traces instead of answer-only public labels.
- Expand Numina CoT training from 5k to 20k examples if the adapter remains competitive.
- Add per-topic error analysis to identify where PoT or QLoRA helps most.

## Speaker notes

The most promising improvement is self-distillation. We can use the strongest base prompt to generate reasoning traces on the train split, keep only correct and well-formatted traces, and then fine-tune on those traces. This is more useful than training directly on final public answers because it teaches the model the solution process.

---

# Slide 17 - What We Learned

## Slide text

- Evaluation discipline matters as much as model choice.
- Prompt and decoding settings can change performance substantially.
- Backends are not interchangeable when adapter support differs.
- Fine-tuning on final answers alone can teach formatting but not robust reasoning.
- Long reasoning models need diagnostics for truncation, boxed-answer coverage, and runtime.

## Speaker notes

This slide summarizes what we learned. One major lesson is that we should not immediately chase the largest training run. We first need baselines, control groups, the same evaluation split, and matched decoding settings. This is central to both machine learning engineering and experimental design.

---

# Slide 18 - Future Work

## Slide text

- Produce final full-public results for the best prompt, PoT, and QLoRA variants.
- Implement trace-based self-distillation from correct model generations.
- Add safer and more structured Python execution for Program-of-Thought.
- Tune QLoRA hyperparameters:
  - larger rank `r=32`
  - longer sequence length 4096
  - lower learning rate `1e-4`
- Build a final ensemble/selection policy across reasoning, PoT, and adapter outputs.

## Speaker notes

The future direction is to turn the current exploration into a final system. Instead of choosing only one model path, we can select strategies by problem type: some problems are better suited for Python execution, some for pure reasoning, and some may benefit from an adapter. Before final submission, all candidates should be compared under the same evaluation setup.
