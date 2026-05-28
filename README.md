# CSE 151B Competition - Final K1 Inference

This repository contains the reproducible inference pipeline for our final
Gradescope/Kaggle submission. The submitted private result used the K1
configuration from `inference_sc_k1_private.ipynb`, now packaged as the
single entry point `run_inference.run_inference()`.

## Final Method

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Backend: vLLM
- Test-time samples: K=1
- Max generated tokens: 16384
- Temperature: 0.6
- Top-p: 0.95
- Top-k: 20
- Thinking mode: enabled
- Fine-tuned weights: none used in the submitted K1 result
- Test-time tools: none

`run_inference()` loads the model, runs inference on the requested dataset,
selects the model response after boxed-answer extraction/voting logic, and
writes the final submission CSV.

## Hardware and Runtime

- GPU type used: NVIDIA A30, 24 GB VRAM
- Approximate generation time for the submitted K1 private run: about 18-24 A30
  GPU-hours. The actual submission was produced with chunked runs across the
  private set, so wall-clock time depended on how many A30 jobs were running in
  parallel.

## Model Weights

No custom fine-tuned checkpoint is required. The final submitted K1 method uses
the base model from HuggingFace Hub:

```text
Qwen/Qwen3-4B-Thinking-2507
```

vLLM/Transformers download the weights automatically on first use. To control
where weights are cached, set `HF_HOME` before running inference:

```bash
export HF_HOME=/path/to/hf_cache
```

## Environment Setup

Use Python 3.11 and install the A30 environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-a30.txt
```

The pipeline expects a CUDA GPU with enough memory for the vLLM settings above.

## Run Inference

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
  --output-path results/submission_final.csv
```

Output files:

```text
results/submission_final.csv
results/submission_final.raw.jsonl
results/submission_final.metadata.json
```

For a quick smoke run, add `--limit 5`. The submitted private run uses the full
dataset with the default `limit=None`.

## Reproducibility Notes

The default `run_inference()` arguments are the final K1 hyperparameters used
for the submitted result. The raw JSONL checkpoint is written after each chunk,
so interrupted runs can be resumed without manual editing.

## Other Branches And Methods Tried

We used separate branches to isolate method families. The final submitted result
was the K1 private inference path on `main`; the branches below record the
experiments and engineering paths that led to that choice.

| Branch | Method / purpose | What worked | What did not work or why it was not final |
|---|---|---|---|
| `main` | Final cleaned K1 submission path. Base `Qwen/Qwen3-4B-Thinking-2507`, vLLM, thinking mode, K=1, 16K token budget, private shard records kept in `results/submission/`. | This is the path matching our submitted private result. It is simple to verify: one model, one inference function, no external tools or adapters. | It keeps the default path aligned with our submitted K1 result rather than the heavier K>1 retry variants. |
| `model-optimization-v1` | Prompt and vLLM optimization branch. It tested fixed eval sizing, stronger MCQ/free-form prompts, forced thinking mode, long output budgets, chunked prefill, prefix caching, auto-resume, and self-consistency settings such as K=7. | Prompt/vLLM optimization was the strongest short-term direction in public smoke testing. The branch notes a 50-question optimization run at 26/50 overall, and forcing thinking mode improved a 50-question test by about 10 percentage points. | Some results were notebook/runtime state rather than clean final artifacts. Larger K improved robustness but increased runtime and memory pressure, and the final private submission was K1 rather than the K7 test setup. |
| `python_vllm` | Program-of-Thought branch. The model generated Python/SymPy/Numpy code, code was executed with timeouts, failed code was retried with error feedback, then unsolved cases fell back to normal reasoning. | The design handled arithmetic/symbolic questions better in principle and added useful diagnostics (`source=python` vs. `source=reasoning`). It also kept batched vLLM generation for pending questions. | It executes model-generated Python at test time, adding safety, timeout, code-extraction, and formatting failure modes. The final Gradescope path stays pure model inference with no subprocess/tool execution. |
| `yang-test` | QLoRA/SFT branch. It added training/eval scripts, Windows and DSMLP guides, public 80/20 train/dev splitting, NuminaMath-CoT training, adapter evaluation, and result tracking. | The QLoRA pipeline worked end-to-end. Public-smoke training produced adapters, and Numina 5k showed some signal under one fair comparison: base 17/50 vs. Numina adapter 20/50 on a held-out public dev split. | The adapter was not consistently better. A later same-split 4096-token comparison favored the base model: base 26/50 vs. Numina adapter 17/50, with weaker free-form performance. We therefore did not submit an adapter. |
| `final-a30-run-inference` | A30 production-inference branch. It packaged a more general `run_inference.py` with vLLM 0.9.1 checks, K=5 defaults, long token budgets, adaptive retry, health summaries, shard resume, public sweeps, and merge helpers. | This branch produced the most complete single-entry engineering pipeline and clarified stable A30 settings: native vLLM, 32K context, chunked generation, fixed sampling parameters, and metadata outputs. | Its default K=5 + retry configuration was not the K1 configuration used for the submitted result. We kept the single-entry design idea, but reset defaults on `main` to match K1. |
| `Submission-makers-for-collab-and-dslmp` | Private submission runner branch for collaborative execution. It added private-set runner guides, a Colab `run_colab.py`, 16-chunk splitting, quarter/chunk equivalence, and merge validation. | It made full private coverage practical across DSMLP and Colab sessions and documented how to avoid duplicate or missing IDs when merging shards. | It was an execution/distribution branch, not a new modeling method. The runner scripts were more manual than the final `run_inference()` requirement. |
| `Antmnguyen-Submission-runner` | Earlier private submission runner branch. It added the private dataset and a quarter-based private notebook/merge guide. | Quarter-based private inference and merge validation worked as the first private-run coordination mechanism. | It was superseded by the more flexible chunk/Colab runner branch and by the final single-entry `run_inference.py`. |

Additional baseline artifacts from earlier history:

- Classical MCQ-only baselines were tested as sanity checks: majority label
  10/75, bag-of-words logistic regression 9/75, and bag-of-words MLP 8/75.
  These confirmed that shallow text classifiers were not competitive for the
  math-reasoning task.
- Starter/prompt smoke runs helped validate the scorer and output format but
  were not treated as final accuracy claims because several were small-sample
  checks.

## Repository Contents

| File | Description |
|---|---|
| `run_inference.py` | Single required inference entry point |
| `requirements-a30.txt` | Python dependencies for the A30 vLLM run |
| `inference_sc_k1_private.ipynb` | Notebook wrapper that calls `run_inference()` |
| `judger.py` | Public-set scoring helper from the competition starter code |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with answers |
| `data/private.jsonl` | Private dataset used for submission generation |
| `results/submission/` | Private-run shard records kept for reproducibility tracking |
