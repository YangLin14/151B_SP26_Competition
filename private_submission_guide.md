# Private Inference — What Changed & Submission Merge Guide

## What Changed in `inference_vllm_prompt_python_run_repeat_private.ipynb`

| Cell | Change |
|---|---|
| **Configuration** | `DATA_PATH` → `data/private.jsonl`; removed `RUN_NAME` / `EVAL_LIMIT` (no longer needed) |
| **Quarter Selector** (new cell) | 5 commented lines — uncomment exactly one to pick which slice of private.jsonl to run |
| **Data Loading** | Slices `data` by quarter at runtime using `n_total // 4`; sets `OUTPUT_PATH` automatically |
| **Scoring → Extraction** | Replaced scoring cell with a no-judge extraction loop (`submission` list of `{id, response}`) |
| **Removed** | Accuracy summary, formatting diagnostics, debug cell — all meaningless without ground truth |
| **Save** | Writes `results/private_{QUARTER}.csv` in `id,response` CSV format matching `sample_submission.csv` |

### Quarter selector (the 5 lines)

```python
QUARTER = "full"  # ALL questions  (100%)      →  results/private_full.csv
# QUARTER = "q1"  # 1st quarter   (first 25%)  →  results/private_q1.csv
# QUARTER = "q2"  # 2nd quarter   (25% – 50%)  →  results/private_q2.csv
# QUARTER = "q3"  # 3rd quarter   (50% – 75%)  →  results/private_q3.csv
# QUARTER = "q4"  # 4th quarter   (last 25%)   →  results/private_q4.csv
```

Uncomment one line, run the full notebook, then switch to the next quarter on another session/GPU.

---

## Merging Four Quarter Files into One Submission CSV

After running all four quarters you will have:

```
results/
  private_q1.csv   ← first 25% of private.jsonl
  private_q2.csv   ← 25% – 50%
  private_q3.csv   ← 50% – 75%
  private_q4.csv   ← last 25%
```

### Requirements the merged file must satisfy

- Header row: `id,response`
- One row per question — **every `id` in `private.jsonl` must appear exactly once**
- `response` is the **full model output** (chain-of-thought + final answer), not just the extracted answer
- Responses are properly CSV-quoted: fields containing commas, newlines, or double quotes are wrapped in double quotes; inner double quotes are escaped as `""`

### Merge script

Save this as `merge_submission.py` in the repo root and run it once after all four quarters finish:

```python
import csv
import json
from pathlib import Path

PRIVATE_JSONL = "data/private.jsonl"
QUARTER_FILES = [
    "results/private_q1.csv",
    "results/private_q2.csv",
    "results/private_q3.csv",
    "results/private_q4.csv",
]
OUTPUT_CSV = "results/submission_final.csv"

# ── Load all ids that must appear in the submission ────────────────────────────
required_ids = {
    json.loads(line)["id"]
    for line in open(PRIVATE_JSONL, encoding="utf-8")
}
print(f"private.jsonl contains {len(required_ids)} questions")

# ── Read all quarter CSVs ──────────────────────────────────────────────────────
rows = {}
for fpath in QUARTER_FILES:
    p = Path(fpath)
    if not p.exists():
        print(f"  [MISSING] {fpath} — skipping")
        continue
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = int(row["id"])
            if qid in rows:
                print(f"  [DUPLICATE] id={qid} found in {fpath} — keeping first occurrence")
                continue
            rows[qid] = row["response"]
    print(f"  [OK] {fpath}  ({sum(1 for _ in open(p, encoding='utf-8')) - 1} rows)")

# ── Validate ───────────────────────────────────────────────────────────────────
missing = required_ids - set(rows.keys())
extra   = set(rows.keys()) - required_ids

if missing:
    print(f"\n[ERROR] {len(missing)} ids are missing from the merged output:")
    print(sorted(missing)[:20], "..." if len(missing) > 20 else "")
if extra:
    print(f"\n[WARNING] {len(extra)} ids in CSVs are not in private.jsonl:")
    print(sorted(extra)[:20])

if missing:
    raise SystemExit("Fix missing ids before submitting.")

# ── Write final CSV sorted by id ──────────────────────────────────────────────
out_path = Path(OUTPUT_CSV)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)   # always quote every field
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
| Missing quarter file | Warns and skips — you will see `[MISSING]` and the validation will fail if ids are absent |
| Duplicate id across files | Keeps the first occurrence, warns |
| Missing id (not in private.jsonl) | Raises and aborts — do not submit until fixed |
| Extra id (in CSV but not in private.jsonl) | Warns but does not abort |

### Output

`results/submission_final.csv` — rows sorted by `id`, every field fully quoted with `csv.QUOTE_ALL`, inner double quotes escaped as `""` automatically by Python's `csv` module. Submit this file.

---

## Quick checklist before submitting

- [ ] All four quarters ran to completion without errors
- [ ] `merge_submission.py` printed `Ready to submit.` with no `[ERROR]` lines
- [ ] Row count in `submission_final.csv` equals the number of lines in `private.jsonl`
- [ ] Spot-check a few rows: `response` should contain the full model output including `\boxed{}`
