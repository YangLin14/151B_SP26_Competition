#!/usr/bin/env python3
import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Optional
import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

# --- 1. Hardware Check Environment Setup ---
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    raise RuntimeError("CUDA is not available. You are not in a GPU pod/session.")

print("=== STARTING HARDWARE FUNCTIONALITY CHECK ===")
try:
    print("Allocating test matrices on GPU (cuda:0)...")
    matrix_a = torch.randn(1000, 1000, device="cuda")
    matrix_b = torch.randn(1000, 1000, device="cuda")

    print("Executing matrix multiplication CUDA kernels...")
    result_matrix = torch.matmul(matrix_a, matrix_b)

    torch.cuda.synchronize()
    
    print("\n[SUCCESS] Pipeline is 100% operational!")
    print(f"-> Verified: CUDA is executing operations successfully.")
    print(f"-> Output Tensor Shape: {result_matrix.shape}")
    print("-> Status: You can safely ignore the architecture warning. Your GPU is active and ready.")
except Exception as error:
    print("\n[FAILURE] Hardware check failed. See error details below:")
    print(str(error))
print("=============================================")

# --- 2. Configuration ---
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/public.jsonl"
RUN_NAME = "prompt_v2_greedy_smoke_50"
OUTPUT_PATH = f"results/{RUN_NAME}.jsonl"
MAX_TOKENS = 32768 
EVAL_LIMIT = 50

print("MODEL_ID:", MODEL_ID)
print("DATA_PATH:", DATA_PATH)
print("RUN_NAME:", RUN_NAME)
print("OUTPUT_PATH:", OUTPUT_PATH)
print("MAX_TOKENS:", MAX_TOKENS)
print("EVAL_LIMIT:", EVAL_LIMIT)

# --- 3. Load the Dataset ---
data_path = Path(DATA_PATH)
assert data_path.exists(), f"Cannot find {DATA_PATH}. Run this notebook from the competition repo root."
data = [json.loads(line) for line in open(data_path, encoding="utf-8")]

if EVAL_LIMIT is None:
    eval_data = data
else:
    eval_data = data[:EVAL_LIMIT]

# --- AUTO-RESUME STATE DETECTOR ---
start_idx = 0
out_path = Path(OUTPUT_PATH)
existing_results = []

if out_path.exists():
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_results.append(json.loads(line))
        start_idx = len(existing_results)
        print(f"\n🔄 [STATE FOUND] Found {start_idx} existing progress records on disk.")
    except Exception as e:
        print(f"⚠️ Error checking progress log file ({str(e)}). Defaulting to starting fresh.")
        start_idx = 0
        existing_results = []

if start_idx >= len(eval_data):
    print(f"✅ [COMPLETE] All {len(eval_data)} evaluations are already processed on disk. Exiting safely.")
    sys.exit(0)

# Slice the remaining workload dynamically
active_eval_data = eval_data[start_idx:]

n_mcq_all  = sum(bool(d.get("options")) for d in data)
n_free_all = sum(not d.get("options") for d in data)
n_mcq_eval  = sum(bool(d.get("options")) for d in eval_data)
n_free_eval = sum(not d.get("options") for d in eval_data)

print(f"Loaded {len(data)} total questions  ({n_mcq_all} MCQ, {n_free_all} free-form)")
print(f"Total Evaluation Target Scope: {len(eval_data)} questions")
print(f"🔥 Active processing target for this run instance: {len(active_eval_data)} remaining questions (Skipping first {start_idx})")

# --- 4. Prompt Construction ---
SYSTEM_PROMPT_FREEFORM = """
Please reason step by step, and put your final answer within \\boxed{}.
If the problem has multiple [ANS] blanks, put the answers in order, separated by commas inside a single \\boxed{}.
""".strip()

SYSTEM_PROMPT_MCQ = """
Please reason step by step, and put your final answer within \\boxed{}.
Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".
""".strip()

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {str(option).strip()}"
            for label, option in zip(labels, options)
        )
        user_prompt = f"""Problem:{question}\nAnswer choices:\n{opts_text}\nSolve the problem and end with the required boxed letter.""".strip()
        return SYSTEM_PROMPT_MCQ, user_prompt

    user_prompt = f"""Problem:{question}\nSolve the problem and end with the required boxed answer.""".strip()
    return SYSTEM_PROMPT_FREEFORM, user_prompt

# --- 5. Load Model with vLLM ---
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
print("Has all_special_tokens_extended:", hasattr(tokenizer, "all_special_tokens_extended"))

vllm_model = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.92,    
    max_model_len=32768,            
    max_num_seqs=16,              
    max_num_batched_tokens=65536,  
    enable_chunked_prefill=True,    
    enable_prefix_caching=True,     
)

sampling_params_sc = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,          
    top_p=0.95,               
    top_k=20,
    n=7,                      
    presence_penalty=1.0,     
    repetition_penalty=1.0,
)
print("Model loaded.")

# --- 6. Generate Responses ---
def format_chat_prompt(item: dict) -> str:
    system, user = build_prompt(item["question"], item.get("options"))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

# Notice we only process active_eval_data here to save vLLM execution cycles!
prompts = [format_chat_prompt(item) for item in active_eval_data]
print(f"Built {len(prompts)} active prompts. K={sampling_params_sc.n}")
print(f"Generating {len(prompts) * sampling_params_sc.n} total samples...")

outputs = vllm_model.generate(prompts, sampling_params=sampling_params_sc)

per_question_raw = []
for out in outputs:
    samples = []
    for o in out.outputs:
        samples.append({
            "text": o.text.strip(),
            "n_tokens": len(o.token_ids),
            "finish_reason": o.finish_reason,   
        })
    per_question_raw.append(samples)

assert len(per_question_raw) == len(active_eval_data)
K = len(per_question_raw[0]) if per_question_raw else sampling_params_sc.n
print(f"\nGeneration complete. K={K}")

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

# --- 7. Score Responses ---
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

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

# Re-hydrate results array with historical state data so final calculation structures remain 100% accurate
results = list(existing_results)

out_path.parent.mkdir(parents=True, exist_ok=True)

# Use append mode ("a") to cleanly append text lines to the log file structure
print(f"Beginning incremental scoring processing loop. Appending updates to {out_path}...")
with open(out_path, "a", encoding="utf-8") as out_file:
    for item, samples in tqdm(zip(active_eval_data, per_question_raw), total=len(active_eval_data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item.get("answer", None)

        sample_texts = [s["text"] for s in samples]
        sample_boxed = [extract_boxed(t) for t in sample_texts]

        if K == 1:
            voted, vote_status = sample_boxed[0], "k1"
        else:
            voted, vote_status = majority_vote(sample_boxed)

        if voted is not None:
            rep_idx = next((i for i, b in enumerate(sample_boxed) if b == voted), 0)
        else:
            rep_idx = 0
        rep_text = sample_texts[rep_idx]

        if gold is None:
            correct = None
        elif is_mcq:
            if voted is not None:
                m = re.search(r"\b([A-Z])\b", voted.strip().upper())
                pred_letter = m.group(1) if m else extract_letter(rep_text)
            else:
                pred_letter = extract_letter(rep_text)
            correct = (pred_letter == str(gold).strip().upper())
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=rep_text,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        has_boxed_per  = [b is not None for b in sample_boxed]
        truncated_per  = [s["finish_reason"] == "length" for s in samples]
        n_tokens_per   = [s["n_tokens"] for s in samples]

        question_record = {
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "gold": gold,
            "K": K,
            "samples_boxed": sample_boxed,
            "voted": voted,
            "vote_status": vote_status,
            "rep_response": rep_text,
            "correct": correct,
            "any_has_boxed":  any(has_boxed_per),
            "all_have_boxed": all(has_boxed_per),
            "any_truncated":  any(truncated_per),
            "all_truncated":  all(truncated_per),
            "tokens_per_sample": n_tokens_per,
            "max_tokens_used": max(n_tokens_per),
        }
        
        results.append(question_record)
        
        # Real-time state syncing with absolute buffer clearing
        out_file.write(json.dumps(question_record, ensure_ascii=False) + "\n")
        out_file.flush()

print(f"Scoring complete. All {len(results)} overall dataset results synchronized on disk.")

# --- 8. Summary ---
scored_results = [r for r in results if r["correct"] is not None]
mcq_res  = [r for r in scored_results if r["is_mcq"]]
free_res = [r for r in scored_results if not r["is_mcq"]]

def acc(subset):
    return sum(bool(r["correct"]) for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 60)
print("EVALUATION RESULTS (COMBINED PROGRESS)")
print("RUN_NAME:", RUN_NAME)
print("=" * 60)
print(f"  MCQ        : {sum(bool(r['correct']) for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(bool(r['correct']) for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(bool(r['correct']) for r in scored_results):4d} / {len(scored_results):4d}  ({acc(scored_results):.2f}%)")
print("=" * 60)

def format_diagnostics(results):
    n = len(results)
    K = results[0]["K"] if results else 0
    has_any   = sum(1 for r in results if r["any_has_boxed"])
    has_all   = sum(1 for r in results if r["all_have_boxed"])
    miss_all  = sum(1 for r in results if not r["any_has_boxed"])
    trunc_any = sum(1 for r in results if r["any_truncated"])
    trunc_all = sum(1 for r in results if r["all_truncated"])
    ties      = sum(1 for r in results if r.get("vote_status") == "tie_first")
    none_vote = sum(1 for r in results if r.get("vote_status") == "all_none")
    avg_tok   = sum(sum(r["tokens_per_sample"]) / len(r["tokens_per_sample"]) for r in results) / n
    pct = lambda x: f"{x}/{n} ({x/n*100:.1f}%)"
    return {
        "RUN_NAME": RUN_NAME,
        "n": n, "K": K,
        "Has Boxed (any sample)":  pct(has_any),
        "Has Boxed (all samples)": pct(has_all),
        "Missing Boxed (all)":     pct(miss_all),
        "Truncated (any sample)":  pct(trunc_any),
        "Truncated (all samples)": pct(trunc_all),
        "Vote ties (K>1 only)":    pct(ties)     if K > 1 else "n/a",
        "All-None votes":          pct(none_vote) if K > 1 else "n/a",
        "Avg tokens/sample":       round(avg_tok, 1),
    }

diag = format_diagnostics(results)
print("=" * 70)
print("FORMATTING DIAGNOSTICS")
print("=" * 70)
for k, v in diag.items():
    print(f"  {k:30s} : {v}")
print("=" * 70)

print("len(data):", len(data))
print("len(eval_data):", len(eval_data))
print("len(results):", len(results))

wrong = [r for r in results if r["correct"] is False]
print("wrong count:", len(wrong))
for r in wrong[:5]:
    print("=" * 100)
    print("id:", r["id"], "is_mcq:", r["is_mcq"], "gold:", r["gold"], "boxed:", r["voted"])
    print("response tail:")
    print(r["rep_response"][-1200:])

print(f"\n[FINAL CHECK] Run evaluation instance cleanly completed. Total logged records: {len(results)}")