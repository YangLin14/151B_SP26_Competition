# Inference Changes: Thinking Mode + Program of Thought

## 1. Force Thinking Mode (`enable_thinking=True`)

Added `enable_thinking=True` to every `apply_chat_template` call.

```python
tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,   # forces <think>...</think> before answer
)
```

Qwen3 thinking models support a `/think` / `/no_think` switch baked into their chat template. Without this flag the model may skip reasoning and answer directly. With it, the model always emits a `<think>...</think>` block before the visible answer. `extract_boxed` uses `rfind` so it correctly picks up the final `\boxed{}` after the thinking block.

---

## 2. Program of Thought (PoT) Pipeline

Replaced the single batched `vllm_model.generate()` call with a 3-phase pipeline:

### Thinking mode is active on the Python path (verified)

`format_python_prompt` passes `enable_thinking=True` to `apply_chat_template`, identical to the reasoning fallback. The model therefore thinks inside `<think>...</think>` tags first — reasoning through the math, choosing the right sympy/numpy approach, planning edge cases — and then writes the `\`\`\`python\`\`\`` block outside those tags. `extract_python_code` searches the full response text with `re.DOTALL` so it correctly finds the code block that appears after the thinking section. No changes were needed here.

### Phase 1 — Python code generation (primary path)
- New system prompts (`SYSTEM_PROMPT_PYTHON_FREEFORM`, `SYSTEM_PROMPT_PYTHON_MCQ`) instruct the model to write a self-contained Python script using `sympy`/`numpy`/`math` and `print()` only the final answer.
- Generated with `sampling_params_python`: `temperature=0.2`, `n=1` (deterministic code).
- Code is extracted from the ` ```python ``` ` block and executed in a subprocess with a 15s timeout.
- On success, stdout is wrapped as `\boxed{<output>}` and stored — the scoring cell requires no changes.

### Phase 2 — Retry with error feedback (1 retry)
- If execution fails, the error message + broken code are fed back into a retry prompt.
- The model sees exactly what went wrong and rewrites the script.
- Controlled by `MAX_PYTHON_RETRIES = 1`.

### Phase 3 — Reasoning fallback
- Questions that still fail after all Python retries fall back to the original thinking-mode reasoning path.
- Generates `n=1` sample (changed from `n=5` — see below).
- All batched, so GPU throughput is preserved.

### Execution stays batched
Each phase calls `vllm_model.generate()` on all pending questions at once — no per-question sequential GPU calls.

---

## 3. Sampling Params: `n=5` → `n=1`

```python
# Before
sampling_params_sc = SamplingParams(..., n=5, ...)

# After
sampling_params_sc = SamplingParams(..., n=1, ...)
```

`n=5` self-consistency was from the original design before PoT existed. The reasoning path is now a last resort for problems where Python fails; running 5 samples there is expensive with little payoff.

---

## 4. Scoring & Diagnostics Updates

- Scoring cell switches from `if K == 1` (global) to `if len(samples) == 1` (per-question), since Python questions have 1 sample and reasoning fallback questions have 1 sample too, but they arrive in the same `per_question_raw` list.
- `"source"` field added to every result (`"python"` or `"reasoning"`).
- Diagnostics cell now reports **Python path** and **Reasoning fallback** counts alongside the existing formatting stats.

---

## Summary Table

| Change | Where | Why |
|---|---|---|
| `enable_thinking=True` | `apply_chat_template` calls | Forces `<think>` reasoning block before every answer |
| Python system prompts + `build_python_prompt` | New cell after prompt cell | Primary PoT path |
| `extract_python_code` + `execute_python` | New cell after model load | Parse and sandbox-execute generated code |
| PoT pipeline loop | Replaced generation cell | Python → retry → reasoning fallback, all batched |
| `n=5` → `n=1` in `sampling_params_sc` | Model load cell | Fallback is last resort; K=5 there is wasteful |
| `"source"` in results + diagnostics | Scoring + diagnostics cells | Track what fraction Python solved vs fell back |

---

## 5. Prompt & Sampling Optimization (Qwen TIR Alignment)

Based on research into the official Qwen2.5-Math TIR training distribution and Qwen3-Thinking model card guidance.

### Python system prompts — TIR canonical phrase (`ccb3e805`)
**Before:** `"Solve this math problem by writing a self-contained Python script."`  
**After:** `"Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \boxed{}."`

The phrase "Please integrate natural language reasoning with programs" appears verbatim in the Qwen2.5-Math HuggingFace model card, GitHub README, and arXiv technical report as the official TIR trigger. Using it aligns the prompt with the token sequences the model saw during post-training SFT/RL, reducing internal prompt conflict when `enable_thinking=True` is active.

### Retry prompt — format reminder added (`ccb3e805`)
Added `"(for multiple answers, comma-separated on one line)"` to the retry user message so a rewrite after failure doesn't silently drift the output format.

### Reasoning fallback prompts — cleaner phrasing (`4e5169ac`)
- `SYSTEM_PROMPT_MCQ`: removed `"answer": "C"` JSON-like phrasing (not in Qwen training distribution). Replaced with `"Your boxed answer must contain exactly one capital letter, e.g. \boxed{C}."` — format the model natively expects.
- `SYSTEM_PROMPT_FREEFORM`: replaced `"[ANS] blanks"` (dataset-specific jargon) with `"fill-in-the-blank placeholders"` and added a concrete example `\boxed{3, 7}` so the model doesn't have to infer the format.

### `presence_penalty=1.0` removed from `sampling_params_sc` (`4b1492d0`)
Presence penalty penalizes tokens that have already appeared, which can prematurely cut the `<think>` block or cause the model to avoid repeating necessary math notation. The official Qwen3 recommended parameter set is temperature / top_p / top_k / repetition_penalty only — presence penalty is not included.

### `sampling_params_python` temperature `0.2` → `0.6` (`4915f406`)
The Qwen3-Thinking model card explicitly warns against temperatures below 0.6 with thinking enabled, stating it causes performance degradation and repetition loops. The thinking block is generated at the same temperature as the code, so 0.2 was below spec. Correctness is guaranteed by subprocess execution and the retry loop, not by temperature.

### `sampling_params_python` max_tokens `4096` → `8192` (`4915f406`)
With `enable_thinking=True`, the `<think>` block on a hard math problem can consume 1,000–2,000 tokens before the model writes any code. At 4096 total tokens, complex problems were at risk of running out of budget before the ` ```python ``` ` block appeared.
