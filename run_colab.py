import os

# T4 (Google Colab free GPU) is sm_75 — override the arch list accordingly
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import re
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

print("Python executable:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
print("CUDA visible device count:", torch.cuda.device_count())
print("CUDA version:", torch.version.cuda)
print("Torch version:", torch.__version__)
print("transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
else:
    raise RuntimeError("CUDA is not available.")

# ── Hardware check ─────────────────────────────────────────────────────────────
print("\n=== STARTING HARDWARE FUNCTIONALITY CHECK ===")
try:
    print("Allocating test matrices on GPU (cuda:0)...")
    matrix_a = torch.randn(1000, 1000, device="cuda")
    matrix_b = torch.randn(1000, 1000, device="cuda")
    print("Executing matrix multiplication CUDA kernels...")
    result_matrix = torch.matmul(matrix_a, matrix_b)
    torch.cuda.synchronize()
    print("\n[SUCCESS] Pipeline is 100% operational!")
    print(f"-> Output Tensor Shape: {result_matrix.shape}")
except Exception as error:
    print("\n[FAILURE] Hardware check failed:", str(error))
print("=============================================\n")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/private.jsonl"
MAX_TOKENS = 16384

print("MODEL_ID:", MODEL_ID)
print("DATA_PATH:", DATA_PATH)
print("MAX_TOKENS:", MAX_TOKENS)

# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK SELECTOR — uncomment EXACTLY ONE line before running
# Each chunk is 1/16 of private.jsonl (~70 questions, ~2–5 hrs on free T4)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Line               Runs on                          Saves to
#  ───────────────    ──────────────────────────────   ─────────────────────────
CHUNK = "full"    # ALL questions  (100%)           →  results/private_full.csv
# CHUNK = "c01"   # Chunk  1 of 16  (first ~6.25%)  →  results/private_c01.csv
# CHUNK = "c02"   # Chunk  2 of 16                   →  results/private_c02.csv
# CHUNK = "c03"   # Chunk  3 of 16                   →  results/private_c03.csv
# CHUNK = "c04"   # Chunk  4 of 16                   →  results/private_c04.csv
# CHUNK = "c05"   # Chunk  5 of 16                   →  results/private_c05.csv
# CHUNK = "c06"   # Chunk  6 of 16                   →  results/private_c06.csv
# CHUNK = "c07"   # Chunk  7 of 16                   →  results/private_c07.csv
# CHUNK = "c08"   # Chunk  8 of 16  (middle)          →  results/private_c08.csv
# CHUNK = "c09"   # Chunk  9 of 16                   →  results/private_c09.csv
# CHUNK = "c10"   # Chunk 10 of 16                   →  results/private_c10.csv
# CHUNK = "c11"   # Chunk 11 of 16                   →  results/private_c11.csv
# CHUNK = "c12"   # Chunk 12 of 16                   →  results/private_c12.csv
# CHUNK = "c13"   # Chunk 13 of 16                   →  results/private_c13.csv
# CHUNK = "c14"   # Chunk 14 of 16                   →  results/private_c14.csv
# CHUNK = "c15"   # Chunk 15 of 16                   →  results/private_c15.csv
# CHUNK = "c16"   # Chunk 16 of 16  (last ~6.25%)   →  results/private_c16.csv

print("CHUNK:", CHUNK)

# ── Load dataset ───────────────────────────────────────────────────────────────
data_path = Path(DATA_PATH)
assert data_path.exists(), f"Cannot find {DATA_PATH}. Run from the competition repo root."

data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
n_total = len(data)

# Chunk boundaries derived by subdividing each quarter into 4 equal parts,
# so c01+c02+c03+c04 = q1, c05+c06+c07+c08 = q2, etc. — matches the A30 notebook exactly.
q = n_total // 4
_q = [0, q, 2 * q, 3 * q, n_total]   # quarter boundary indices (same formula as A30 notebook)

_slices = {"full": (0, n_total)}
for qi in range(4):
    qs, qe = _q[qi], _q[qi + 1]
    sub = (qe - qs) // 4
    for ci in range(4):
        name = f"c{qi * 4 + ci + 1:02d}"
        cs = qs + ci * sub
        ce = qs + (ci + 1) * sub if ci < 3 else qe   # last sub-chunk takes remainder
        _slices[name] = (cs, ce)

start, end = _slices[CHUNK]
eval_data   = data[start:end]
OUTPUT_PATH = f"results/private_{CHUNK}.csv"

print(f"Loaded {n_total} total questions from private.jsonl")
print(f"CHUNK '{CHUNK}': indices [{start}, {end})  →  {len(eval_data)} questions")
print(f"OUTPUT_PATH: {OUTPUT_PATH}")

# ── Reasoning system prompts ───────────────────────────────────────────────────
SYSTEM_PROMPT_FREEFORM = """Please reason step by step, and put your final answer within \\boxed{}.
If the problem asks for multiple values or has multiple fill-in-the-blank placeholders, list all answers in order inside a single \\boxed{}, separated by commas, e.g. \\boxed{3, 7}.""".strip()

SYSTEM_PROMPT_MCQ = """Please reason step by step, and put your final answer within \\boxed{}.
Your boxed answer must contain exactly one capital letter representing the correct choice, e.g. \\boxed{C}.""".strip()


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {str(option).strip()}" for label, option in zip(labels, options))
        user_prompt = f"Problem:\n{question}\n\nAnswer choices:\n{opts_text}\n\nSolve the problem and end with the required boxed letter."
        return SYSTEM_PROMPT_MCQ, user_prompt
    user_prompt = f"Problem:\n{question}\n\nSolve the problem and end with the required boxed answer."
    return SYSTEM_PROMPT_FREEFORM, user_prompt


# ── Python (Program of Thought) system prompts ────────────────────────────────
SYSTEM_PROMPT_PYTHON_FREEFORM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.

Write a self-contained Python script to compute the answer:
1. Use sympy for symbolic/exact answers; numpy or math for numerical computation
2. Your script's LAST print() must output ONLY the answer value — no labels, no units, no extra text
3. If the problem has multiple fill-in-the-blank placeholders, print all answers comma-separated on one line
4. Wrap your code in ```python ... ``` blocks""".strip()

SYSTEM_PROMPT_PYTHON_MCQ = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.

Write a self-contained Python script to derive the answer and identify the matching choice:
1. Use sympy for exact symbolic computation
2. Your script's LAST print() must output ONLY the single capital letter of the correct choice (e.g. C)
3. Wrap your code in ```python ... ``` blocks""".strip()


def build_python_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {str(opt).strip()}" for label, opt in zip(labels, options))
        user_prompt = (
            f"Problem:\n{question}\n\nAnswer choices:\n{opts_text}\n\n"
            "Write Python code to solve this. The last print() must output only the correct letter."
        )
        return SYSTEM_PROMPT_PYTHON_MCQ, user_prompt
    user_prompt = (
        f"Problem:\n{question}\n\n"
        "Write Python code to solve this. The last print() must output only the final answer."
    )
    return SYSTEM_PROMPT_PYTHON_FREEFORM, user_prompt


def build_python_retry_prompt(question: str, options: Optional[list], prev_code: str, error: str) -> tuple[str, str]:
    sys_p, _ = build_python_prompt(question, options)
    user_prompt = (
        f"Problem:\n{question}\n\n"
        f"Your previous Python attempt failed with this error:\n{error}\n\n"
        f"Failed code:\n```python\n{prev_code}\n```\n\n"
        "Fix the error and write a correct Python solution. "
        "The last print() must output only the final answer "
        "(for multiple answers, comma-separated on one line)."
    )
    return sys_p, user_prompt


print("Prompts loaded.")

# ── Load tokenizer ─────────────────────────────────────────────────────────────
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
    print("Patching Qwen2Tokenizer.all_special_tokens_extended ...")

    @property
    def all_special_tokens_extended(self):
        return list(self.all_special_tokens)

    Qwen2Tokenizer.all_special_tokens_extended = all_special_tokens_extended
else:
    print("Qwen2Tokenizer already has all_special_tokens_extended.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="left",
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer class:", tokenizer.__class__)

# ── Load vLLM model ────────────────────────────────────────────────────────────
# GPU changes vs A30 version (T4 Colab free tier, 15 GB VRAM):
#   dtype          : bfloat16 → float16   (T4/sm_75 has no bfloat16 tensor cores)
#   gpu_memory_utilization: 0.92 → 0.90  (small safety buffer on 15 GB)
#   max_num_seqs   : 16 → 1              (15 GB can't hold 16 full KV caches; speed doesn't matter)
#   max_num_batched_tokens: 32768 → 32768 (unchanged; already correct for 1 seq × 32768 tokens)
# All inference parameters (MAX_TOKENS, max_model_len, sampling) are UNCHANGED.
vllm_model = LLM(
    model=MODEL_ID,
    dtype="float16",                  # T4 does not support bfloat16
    trust_remote_code=True,
    gpu_memory_utilization=0.90,      # 15 GB × 0.90 ≈ 13.5 GB budget
    max_model_len=32768,              # unchanged — full context preserved
    max_num_seqs=1,                   # one sequence at a time fits T4 KV cache; speed doesn't matter
    max_num_batched_tokens=32768,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
)

sampling_params_sc = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    n=1,
    repetition_penalty=1.0,
)

print("Model loaded.")

# ── Python execution utilities ─────────────────────────────────────────────────
def extract_python_code(text: str) -> Optional[str]:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if 'print' in code:
            return code
    return None


def execute_python(code: str, timeout: int = 15) -> tuple[Optional[str], Optional[str]]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        os.unlink(tmp_path)
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return (out, None) if out else (None, "Script produced no output.")
        return None, proc.stderr.strip()[-500:]
    except subprocess.TimeoutExpired:
        try: os.unlink(tmp_path)
        except: pass
        return None, f"Timed out after {timeout}s."
    except Exception as e:
        try: os.unlink(tmp_path)
        except: pass
        return None, str(e)


print("Python execution utilities loaded.")

# ── PoT Generation Pipeline ────────────────────────────────────────────────────
PYTHON_TIMEOUT     = 15
MAX_PYTHON_RETRIES = 2
MAX_CODE_RESTARTS  = 3


def format_chat_prompt(item: dict) -> str:
    system, user = build_prompt(item["question"], item.get("options"))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def format_python_prompt(item: dict, prev_code: str = None, error: str = None) -> str:
    if prev_code is not None and error is not None:
        system, user = build_python_retry_prompt(item["question"], item.get("options"), prev_code, error)
    else:
        system, user = build_python_prompt(item["question"], item.get("options"))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


sampling_params_python = SamplingParams(
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    n=1,
    repetition_penalty=1.0,
)

pending   = list(range(len(eval_data)))
py_state  = {}
final_raw = {}

for restart in range(MAX_CODE_RESTARTS):
    if not pending:
        break

    print(f"\n{'═'*60}")
    print(f"Code restart {restart + 1}/{MAX_CODE_RESTARTS}  ({len(pending)} questions pending)")
    print(f"{'═'*60}")

    for idx in pending:
        py_state.pop(idx, None)

    for attempt in range(MAX_PYTHON_RETRIES + 1):
        if not pending:
            break

        print(f"\n── Python attempt {attempt + 1}/{MAX_PYTHON_RETRIES + 1}  ({len(pending)} questions) ──")

        py_prompts = [
            format_python_prompt(
                eval_data[idx],
                prev_code=py_state.get(idx, {}).get("code"),
                error=py_state.get(idx, {}).get("error"),
            )
            for idx in pending
        ]
        py_outputs = vllm_model.generate(py_prompts, sampling_params=sampling_params_python)

        still_pending = []
        for idx, out in zip(pending, py_outputs):
            resp  = out.outputs[0].text.strip()
            n_tok = len(out.outputs[0].token_ids)
            code  = extract_python_code(resp)

            if code is None:
                py_state[idx] = {"code": "", "error": "No ```python``` block found in response."}
                still_pending.append(idx)
                continue

            stdout, err = execute_python(code, timeout=PYTHON_TIMEOUT)

            if stdout is not None:
                final_raw[idx] = [{
                    "text": f"\\boxed{{{stdout.strip()}}}",
                    "n_tokens": n_tok,
                    "finish_reason": "stop",
                    "source": "python",
                }]
            else:
                py_state[idx] = {"code": code, "error": err}
                still_pending.append(idx)

        pending = still_pending
        print(f"   Solved so far: {len(final_raw)}/{len(eval_data)}  |  still pending: {len(pending)}")

if pending:
    print(f"\n── Reasoning fallback for {len(pending)} questions (all {MAX_CODE_RESTARTS} code cycles failed) ──")
    fb_prompts = [format_chat_prompt(eval_data[idx]) for idx in pending]
    fb_outputs = vllm_model.generate(fb_prompts, sampling_params=sampling_params_sc)

    for idx, out in zip(pending, fb_outputs):
        final_raw[idx] = [
            {
                "text": o.text.strip(),
                "n_tokens": len(o.token_ids),
                "finish_reason": o.finish_reason,
                "source": "reasoning",
            }
            for o in out.outputs
        ]

per_question_raw = [final_raw[i] for i in range(len(eval_data))]
assert len(per_question_raw) == len(eval_data)

n_python   = sum(1 for s in per_question_raw if s[0].get("source") == "python")
n_fallback = sum(1 for s in per_question_raw if s[0].get("source") == "reasoning")
print(f"\nPipeline complete.")
print(f"  Python path : {n_python}/{len(eval_data)}")
print(f"  Reasoning   : {n_fallback}/{len(eval_data)}")

# ── Extract boxed ──────────────────────────────────────────────────────────────
def extract_boxed(text: str):
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None


def extract_letter(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"answer is ([A-E])",
        r"answer is: ([A-E])",
        r"answer is \(([A-E])\)",
        r"Choice ([A-E])",
        r"Option ([A-E])",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    last_bit = text[-50:].upper()
    m = re.search(r"\b([A-E])\b", last_bit)
    if m:
        return m.group(1).upper()
    return ""


# ── Extract representative response per question ───────────────────────────────
def majority_vote(boxed_answers):
    valid = [b for b in boxed_answers if b is not None]
    if not valid:
        return None, "all_none"
    counts = {}
    for b in valid:
        counts[b] = counts.get(b, 0) + 1
    max_count = max(counts.values())
    winners = {b for b, c in counts.items() if c == max_count}
    if len(winners) == 1:
        return next(iter(winners)), "majority"
    for b in valid:
        if b in winners:
            return b, "tie_first"


submission = []
for item, samples in zip(eval_data, per_question_raw):
    sample_texts = [s["text"] for s in samples]
    sample_boxed = [extract_boxed(t) for t in sample_texts]

    if len(samples) == 1:
        rep_text = sample_texts[0]
    else:
        voted, _ = majority_vote(sample_boxed)
        rep_idx  = next((i for i, b in enumerate(sample_boxed) if b == voted), 0)
        rep_text = sample_texts[rep_idx]

    submission.append({"id": item["id"], "response": rep_text})

print(f"Extraction complete: {len(submission)} rows")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"])
    writer.writeheader()
    writer.writerows(submission)

print(f"Saved {len(submission)} rows → {out_path}")
print(f"CHUNK: {CHUNK}  |  question IDs: {submission[0]['id']} … {submission[-1]['id']}")
