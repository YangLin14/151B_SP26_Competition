# Google Colab Setup Guide — `run_colab.py`

## Before You Start

**Files you need on your local machine:**
- `run_colab.py` (the inference script)
- `data/private.jsonl` (the private test set)

**Time estimate per chunk:** The dataset is split into **16 chunks** of ~70 questions each. On a free T4 GPU running one sequence at a time, each chunk takes roughly **2–5 hours** — well within a single Colab session. Running all 16 chunks across 16 sessions covers the full dataset.

---

## Step 1 — Open Google Colab

1. In your browser, go to **https://colab.research.google.com**
2. Sign in with your Google account if prompted
3. In the dialog that appears, click **New notebook** (top-left of the dialog, or use **File → New notebook** from the menu bar)

---

## Step 2 — Select the T4 GPU Runtime

1. In the top menu bar, click **Runtime**
2. Click **Change runtime type**
3. A dialog box opens. Under **Hardware accelerator**, click the dropdown (it says "None" by default) and select **T4 GPU**
4. Click **Save**
5. In the top-right corner of the page, click the **Connect** button (it looks like two linked circles). Wait until it turns into a green checkmark with RAM/Disk bars — this means the GPU instance is ready

> If you see a "Connect to a hosted runtime" popup instead of the button, just click that.

---

## Step 3 — Mount Google Drive (model cache — do this first)

The Qwen3-4B model is ~8 GB. Without Drive mounting, it re-downloads from HuggingFace every new session (~10–20 minutes each time). Mounting Drive saves it once permanently.

In a new code cell, paste and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

A popup or inline link will appear asking you to authorize. Click it, choose your Google account, click **Allow**, then copy the authorization code back into the cell and press Enter.

You should see: `Mounted at /content/drive`

---

## Step 4 — Install Dependencies

Paste this into a new code cell and run it. It will take **5–10 minutes**.

```python
!pip install vllm sympy tqdm --quiet
```

> **Important:** After installation finishes, Colab will likely show a yellow warning bar at the top that says **"RESTART RUNTIME"** (because vllm upgrades some pre-installed packages). Click that button. The runtime restarts — this is expected and required.

After the restart, your code cells will show as not yet run (gray). That is correct. Continue to the next step.

---

## Step 5 — Re-mount Google Drive After Restart

Because the runtime restarted, you need to re-run the Drive mount. Paste and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the same authorization steps as before (it may skip auth if your token is still valid).

---

## Step 6 — Point HuggingFace Cache at Google Drive

This makes the model download persist between sessions so you never re-download it.

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.makedirs("/content/drive/MyDrive/hf_cache", exist_ok=True)
print("HF_HOME set to Google Drive. Model will be cached there.")
```

> **Every time you start a new session**, you must re-run Step 5 (mount Drive) and this cell before running the script, or the model will re-download to ephemeral storage.

---

## Step 7 — Create the Required Directories

```python
!mkdir -p /content/data /content/results
```

---

## Step 8 — Upload `run_colab.py`

1. In the **left sidebar**, click the **folder icon** (it looks like a file cabinet) to open the file browser
2. Click the **Upload** button — it is the icon that looks like a page with an upward arrow, at the top of the file browser panel
3. In the file picker that opens, navigate to your local machine and select `run_colab.py`
4. Wait for the upload bar to complete. The file will appear in the file browser under `/content/`

---

## Step 9 — Upload `private.jsonl`

1. In the same file browser, **double-click the `data` folder** to navigate into it
2. Click the **Upload** button again
3. Select `private.jsonl` from your local machine
4. Wait for the upload to finish

Verify the file is in the right place:

```python
!ls /content/data/
# Should print: private.jsonl
```

---

## Step 10 — Select Which Chunk to Run

`run_colab.py` splits the dataset into **16 chunks** (c01–c16). You must edit the CHUNK SELECTOR block to uncomment exactly one line before running. There are two ways to do this:

### Option A — Edit in the file browser (easiest)
1. In the left sidebar file browser, double-click `run_colab.py`
2. The file opens in a text editor tab inside Colab
3. Find the CHUNK SELECTOR block (lines ~65–85). It looks like this:
   ```python
   CHUNK = "full"    # ALL questions  (100%)
   # CHUNK = "c01"   # Chunk  1 of 16  (first ~6.25%)
   # CHUNK = "c02"   # Chunk  2 of 16
   # CHUNK = "c03"   # Chunk  3 of 16
   # ...
   # CHUNK = "c16"   # Chunk 16 of 16  (last ~6.25%)
   ```
4. Comment out `CHUNK = "full"` by adding a `#` at the start of that line
5. Uncomment the chunk you want by removing its leading `# `
6. Press **Ctrl+S** (or **Cmd+S** on Mac) to save

### Option B — Edit via a code cell
```python
# Change "c01" to whichever chunk number you want to run (c01 through c16)
CHUNK_TO_SET = "c01"

with open("/content/run_colab.py", "r") as f:
    src = f.read()

import re
# Comment out all active CHUNK= lines, then uncomment the one we want
src = re.sub(r'^(CHUNK\s*=.*)', r'# \1', src, flags=re.MULTILINE)
src = re.sub(r'^#\s*(CHUNK\s*=\s*["\']' + CHUNK_TO_SET + r'["\'].*)', r'\1', src, flags=re.MULTILINE)

with open("/content/run_colab.py", "w") as f:
    f.write(src)

print(f"CHUNK set to: {CHUNK_TO_SET}")
!grep "^CHUNK" /content/run_colab.py
```

---

## Step 11 — Run the Script

```python
!cd /content && HF_HOME=/content/drive/MyDrive/hf_cache python run_colab.py
```

> The `HF_HOME=...` prefix ensures the model is read from (and saved to) Google Drive even if you forgot to run Step 6.

**What you will see:**
1. GPU info and hardware check
2. `Loaded N total questions` and chunk slice info (e.g., `CHUNK 'c01': indices [0, 70) → 70 questions`)
3. `Fetching model weights...` — the **first run** downloads ~8 GB to Drive (takes 10–20 minutes). All subsequent sessions skip this entirely.
4. `Model loaded.` — model is in VRAM, inference starts
5. Per-question progress: `Code restart 1/3 ... Python attempt 1/3 ...`
6. Final: `Saved 70 rows → results/private_c01.csv`

**Keep the browser tab open and do not let your computer sleep** while this runs. Free Colab disconnects idle sessions; an actively executing cell keeps the session alive.

---

## Step 12 — Download the Results

When the script finishes, download the output CSV:

```python
from google.colab import files
import os

chunk = "c01"  # change to match whichever chunk you ran
path = f"/content/results/private_{chunk}.csv"

if os.path.exists(path):
    files.download(path)
    print(f"Downloaded {path}")
else:
    print("File not found — check the script output for errors")
```

Save the downloaded CSV somewhere safe on your local machine. You will collect all 16 before merging.

---

## Step 13 — Run Remaining Chunks (c02 through c16)

For each remaining chunk, start a **new Colab session** (or reuse the current one if it is still alive):

1. Go back to your notebook at colab.research.google.com
2. Verify you still have a T4 GPU (**Runtime → Change runtime type**)
3. Run **Step 5** (remount Drive) and **Step 6** (set HF_HOME) — the model will NOT re-download since it is cached on Drive
4. Run **Step 7** (`mkdir`) and re-upload `run_colab.py` and `private.jsonl` — the `/content/` directory is wiped on every session end
5. Run **Step 10** (Option A or B) to set the next chunk number
6. Run **Step 11** to start inference
7. Run **Step 12** to download that chunk's CSV

Repeat until you have `private_c01.csv` through `private_c16.csv`.

### Mixing Colab chunks with A30 quarters

Chunks and quarters are exactly interchangeable — 4 chunks = 1 quarter:

| Quarter (A30 notebook) | Chunks (Colab) |
|---|---|
| q1 — first 25% | c01 + c02 + c03 + c04 |
| q2 — second 25% | c05 + c06 + c07 + c08 |
| q3 — third 25% | c09 + c10 + c11 + c12 |
| q4 — last 25% | c13 + c14 + c15 + c16 |

The alignment is guaranteed by the code: `run_colab.py` computes quarter boundaries using the same `n_total // 4` formula as the A30 notebook, then subdivides each quarter into 4 equal sub-chunks. So the question index ranges are identical across both files — no boundary drift from integer division.

You can therefore mix hardware freely — e.g. run q1 and q2 on the A30, then run c09–c16 on Colab for the second half. **Never run both a quarter and its four corresponding chunks** — that would double-cover those questions and produce duplicates.

### Chunk tracking checklist

| Chunk | Questions | Status | File |
|---|---|---|---|
| c01 | ~70 | | private_c01.csv |
| c02 | ~70 | | private_c02.csv |
| c03 | ~70 | | private_c03.csv |
| c04 | ~70 | | private_c04.csv |
| c05 | ~70 | | private_c05.csv |
| c06 | ~70 | | private_c06.csv |
| c07 | ~70 | | private_c07.csv |
| c08 | ~70 | | private_c08.csv |
| c09 | ~70 | | private_c09.csv |
| c10 | ~70 | | private_c10.csv |
| c11 | ~70 | | private_c11.csv |
| c12 | ~70 | | private_c12.csv |
| c13 | ~70 | | private_c13.csv |
| c14 | ~70 | | private_c14.csv |
| c15 | ~70 | | private_c15.csv |
| c16 | ~70 | | private_c16.csv |

---

## Step 14 — Merge Into Final Submission

Once you have all 16 CSV files locally, run `merge_submission.py` from the repo root on your local machine:

```bash
python merge_submission.py
```

This produces `results/submission_final.csv` — see `private_submission_guide.md` for full details on the merge script and validation checks.

---

## Session Cheat Sheet

| Task | Command / Location |
|---|---|
| Select GPU | Runtime → Change runtime type → T4 GPU → Save |
| Mount Drive | `drive.mount('/content/drive')` |
| Set HF cache | `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` |
| Install deps | `!pip install vllm sympy tqdm --quiet` then restart runtime |
| Upload file | Left sidebar → folder icon → upload icon |
| Set chunk | Edit `run_colab.py` or use Option B code cell |
| Run script | `!cd /content && HF_HOME=/content/drive/MyDrive/hf_cache python run_colab.py` |
| Download result | `from google.colab import files; files.download(...)` |

---

## Troubleshooting

**`CUDA out of memory`**
The T4 has 15 GB. The model in float16 uses ~8 GB; the KV cache for one sequence at 32 768 tokens uses ~4.5 GB. Total ≈ 12.5 GB which fits under the 0.90 utilization cap. If you still OOM, try closing any other Colab tabs that are connected to a GPU runtime.

**`vllm not found` after runtime restart**
vllm is installed into the ephemeral session — it does not persist. Re-run the `pip install` cell every new session.

**Model re-downloading despite Google Drive mount**
Make sure you ran Step 5 (remount Drive) AND Step 6 (set HF_HOME) before running the script. If `HF_HOME` is not set, the model downloads to `/root/.cache/huggingface/` which is ephemeral.

**Session disconnected mid-run**
Results written so far are lost (the CSV is only saved at the very end of the script). Restart the same chunk from scratch. To reduce this risk, make sure your laptop/PC does not go to sleep and the Colab browser tab stays open and in focus.

**`No module named 'sympy'`**
The generated Python scripts use sympy internally. Re-run `!pip install sympy`.

**Script runs but outputs `0 rows saved`**
Verify `data/private.jsonl` is at `/content/data/private.jsonl` with `!ls /content/data/`. If the path is wrong the data load assertion will fail immediately.
