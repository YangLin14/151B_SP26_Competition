# Private Inference — What Changed & Submission Merge Guide

## What Changed in `inference_vllm_prompt_python_run_repeat_private.ipynb`

| Cell | Change |
|---|---|
| **Configuration** | `DATA_PATH` → `data/private.jsonl`; removed `RUN_NAME` / `EVAL_LIMIT` (no longer needed) |
| **Chunk Selector** (new cell) | 17 commented lines — uncomment exactly one to pick which slice of private.jsonl to run |
| **Data Loading** | Slices `data` by chunk at runtime using `n_total // 16`; sets `OUTPUT_PATH` automatically |
| **Scoring → Extraction** | Replaced scoring cell with a no-judge extraction loop (`submission` list of `{id, response}`) |
| **Removed** | Accuracy summary, formatting diagnostics, debug cell — all meaningless without ground truth |
| **Save** | Writes `results/private_{CHUNK}.csv` in `id,response` CSV format matching `sample_submission.csv` |

---

## Two Ways to Run — Quarters (A30) vs Chunks (Colab)

The dataset can be split two ways depending on which hardware you use:

| System | File | Selector variable | Split | Files produced |
|---|---|---|---|---|
| **A30 notebook** | `inference_vllm_prompt_python_run_repeat_private.ipynb` | `QUARTER` | 4 × 25% | `private_q1.csv` … `private_q4.csv` |
| **Colab script** | `run_colab.py` | `CHUNK` | 16 × 6.25% | `private_c01.csv` … `private_c16.csv` |

**These are exactly equivalent — 4 chunks = 1 quarter:**

| Quarter (A30) | Chunks (Colab) |
|---|---|
| q1 — first 25% | c01 + c02 + c03 + c04 |
| q2 — second 25% | c05 + c06 + c07 + c08 |
| q3 — third 25% | c09 + c10 + c11 + c12 |
| q4 — last 25% | c13 + c14 + c15 + c16 |

The alignment is guaranteed by the code: `run_colab.py` computes quarter boundaries using the same `n_total // 4` formula as the notebook, then subdivides each quarter into 4 equal sub-chunks. So the question index ranges are identical across both files — no boundary drift from integer division.

You can run the whole dataset on the A30 (4 sessions), on Colab (16 sessions), or mix — e.g. run q1 and q2 on the A30, then run c09–c16 on Colab for the second half. **Never run both a quarter and its four corresponding chunks** — that would double-cover those questions and produce duplicates.

---

## Merging Into One Submission CSV

### Requirements the merged file must satisfy

- Header row: `id,response`
- One row per question — **every `id` in `private.jsonl` must appear exactly once**
- `response` is the **full model output** (chain-of-thought + final answer), not just the extracted answer
- Responses are properly CSV-quoted: fields containing commas, newlines, or double quotes are wrapped in double quotes; inner double quotes are escaped as `""`

### Merge script

Save this as `merge_submission.py` in the repo root. It auto-detects whichever quarter/chunk files are present in `results/` and merges them all by question ID.

```python
import csv
import json
import glob
from pathlib import Path

PRIVATE_JSONL = "data/private.jsonl"
OUTPUT_CSV    = "results/submission_final.csv"

# ── Auto-detect output files (quarter naming q1-q4, chunk naming c01-c16, or mix)
quarter_files = sorted(glob.glob("results/private_q*.csv"))   # A30  — 1/4 slices
chunk_files   = sorted(glob.glob("results/private_c*.csv"))   # Colab — 1/16 slices
all_files     = quarter_files + chunk_files

if not all_files:
    raise SystemExit("No quarter or chunk files found in results/. Run inference first.")

print(f"Found {len(quarter_files)} quarter file(s) and {len(chunk_files)} chunk file(s):")
for f in all_files:
    print(f"  {f}")

# ── Load all ids that must appear in the submission ────────────────────────────
required_ids = {
    json.loads(line)["id"]
    for line in open(PRIVATE_JSONL, encoding="utf-8")
}
print(f"\nprivate.jsonl requires {len(required_ids)} question IDs")

# ── Read and merge all files ───────────────────────────────────────────────────
rows = {}
for fpath in all_files:
    p = Path(fpath)
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            if qid in rows:
                print(f"  [DUPLICATE] id={qid} already seen — found again in {fpath}. Keeping first.")
                continue
            rows[qid] = row["response"]
    n_rows = sum(1 for _ in open(p, encoding="utf-8")) - 1
    print(f"  [OK] {fpath}  ({n_rows} rows, running total: {len(rows)})")

# ── Validate ───────────────────────────────────────────────────────────────────
missing = required_ids - set(rows.keys())
extra   = set(rows.keys()) - required_ids

if missing:
    print(f"\n[ERROR] {len(missing)} IDs from private.jsonl are missing:")
    print("  First 20:", sorted(missing)[:20])
    print("  Tip: check the quarter/chunk equivalence table — you may have a gap.")
if extra:
    print(f"\n[WARNING] {len(extra)} IDs in the CSVs are not in private.jsonl:")
    print("  First 20:", sorted(extra)[:20])

if missing:
    raise SystemExit("Fix missing IDs before submitting.")

# ── Write final CSV sorted by id ──────────────────────────────────────────────
out_path = Path(OUTPUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "response"])
    for qid in sorted(rows.keys()):
        writer.writerow([qid, rows[qid]])

print(f"\nSaved {len(rows)} rows → {out_path}")
print("Ready to submit.")
```

Run it:

```bash
python merge_submission.py
```

### What the script checks

| Check | What happens |
|---|---|
| Missing file | Skips it — validation will catch missing IDs and abort |
| Duplicate ID (overlapping quarter + chunks) | Keeps first occurrence, warns — fix by not double-running the same range |
| Missing ID (gap in coverage) | Prints which IDs are absent and aborts — check the equivalence table above |
| Extra ID (not in private.jsonl) | Warns but does not abort |

### Output

`results/submission_final.csv` — rows sorted by `id`, every field fully quoted with `csv.QUOTE_ALL`, inner double quotes escaped as `""` automatically by Python's `csv` module. Submit this file.

---

## Quick checklist before submitting

- [ ] All runs completed without errors (4 quarters on A30, or 16 chunks on Colab, or a valid mix)
- [ ] No quarter and its 4 corresponding chunks were both run (that causes duplicates)
- [ ] `merge_submission.py` printed `Ready to submit.` with no `[ERROR]` lines
- [ ] Row count in `submission_final.csv` equals the number of lines in `private.jsonl`
- [ ] Spot-check a few rows: `response` should contain the full model output including `\boxed{}`
