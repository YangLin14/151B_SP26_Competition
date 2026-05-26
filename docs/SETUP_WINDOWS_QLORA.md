# Windows QLoRA SFT Setup Notes

This note documents the exact setup and fixes used to run the QLoRA SFT smoke test on a Windows machine with an NVIDIA GPU.

Target repo:

```powershell
C:\Users\es-tec\Downloads\151B_SP26_Competition
```

Working smoke-test command:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 1024
```

---

## 0. Important Windows Warning

Do **not** try to install `vllm` on native Windows.

`vllm` does not properly support native Windows. If a package install tries to build `vllm`, it may fail with errors like:

```text
vLLM only supports Linux platform (including WSL) and MacOS. Building on win32...
```

For Windows native training, use the minimal QLoRA training dependencies instead of installing the full Linux-oriented environment.

If `vllm` is required later, use one of these instead:

```text
WSL2 Ubuntu
Linux server / DSMLP
Colab
```

---

## 1. Install `uv` if Python is not available

If these commands fail:

```powershell
python --version
py --version
```

then install `uv`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
irm https://astral.sh/uv/install.ps1 | iex
```

Close PowerShell, reopen it, then check:

```powershell
uv --version
```

If `uv` is still not recognized, temporarily add it to PATH:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv --version
```

---

## 2. Create the Python 3.11 virtual environment

Go to the repo:

```powershell
cd C:\Users\es-tec\Downloads\151B_SP26_Competition
```

Install Python 3.11 through `uv`:

```powershell
uv python install 3.11
```

Create the virtual environment:

```powershell
uv venv .venv --python 3.11 --seed
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You should see something like this:

```powershell
(.venv) PS C:\Users\es-tec\Downloads\151B_SP26_Competition>
```

Check Python:

```powershell
python --version
```

Expected:

```text
Python 3.11.x
```

---

## 3. Install minimal training dependencies

Do **not** blindly install a full requirements file if it includes `vllm`, `flash-attn`, `triton`, or other Linux-specific packages.

First upgrade basic tools:

```powershell
python -m pip install --upgrade pip setuptools wheel numpy
```

Install PyTorch CUDA wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install QLoRA / SFT training packages:

```powershell
python -m pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf scipy tqdm pandas scikit-learn antlr4-python3-runtime==4.11.1
```

---

## 4. Verify GPU access

Check the NVIDIA driver:

```powershell
nvidia-smi
```

Then check PyTorch CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected result should include:

```text
True
NVIDIA GeForce RTX 3060
```

If it prints `False`, PyTorch was probably installed incorrectly or the CUDA wheel does not match the system.

---

## 5. Fix Windows UTF-8 / TRL import issue

On Windows, `trl` may fail during import with this error:

```text
UnicodeDecodeError: 'charmap' codec can't decode byte 0x81
```

This happens because Windows may default to a non-UTF-8 encoding such as `cp1252`, while TRL reads UTF-8 `.jinja` chat template files.

Fix it in the current PowerShell session:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Then test TRL import:

```powershell
python -X utf8 -c "from trl import SFTTrainer, SFTConfig; print('TRL import OK')"
```

Expected:

```text
TRL import OK
```

Optional permanent fix:

```powershell
setx PYTHONUTF8 1
setx PYTHONIOENCODING utf-8
```

After using `setx`, close PowerShell and reopen it.

---

## 6. Run the QLoRA SFT smoke test

Use a conservative sequence length first. On an RTX 3060, start with `1024`, not `2048`.

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 1024
```

If this works, then try increasing sequence length later:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke_2048 --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 2048
```

If `2048` causes out-of-memory errors, go back to `1024`.

---

## 7. Evaluate on native Windows

Do not use `scripts/qlora_vllm_eval.py` on native Windows because it imports
`vllm`.

Use the Transformers eval script instead. First run the base-model control on
the held-out dev split:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_base_control_transformers_eval_50.jsonl
```

Then run the QLoRA adapter on the same examples and settings:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_public_smoke/final_adapter --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_sft_public_smoke_transformers_eval_50.jsonl
```

The adapter is only useful if it beats the base-model control under the same
eval settings.

If generation finished but scoring failed, install the scoring dependency and
score the saved raw generations without rerunning the model:

```powershell
python -m pip install antlr4-python3-runtime==4.11.1
python -X utf8 scripts/qlora_score_raw.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --raw-path results/qlora_base_control_transformers_eval_10.raw.jsonl --output-path results/qlora_base_control_transformers_eval_10.jsonl
```

---

## 8. NuminaMath CoT smoke training on native Windows

Use the streamed Numina path. The command should print a line like:

```text
Streaming 500 examples from AI-MO/NuminaMath-CoT with shuffle_buffer=10000...
```

PowerShell:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_numina_smoke_1024 --data-source numina --max-train-examples 500 --max-steps 50 --max-seq-len 1024 --numina-shuffle-buffer 10000
```

If you see this instead, you are using an old script or disabled streaming:

```text
Generating train split: 0/859494
```

Stop that run and use the updated script. The selected streamed subset is saved
to:

```text
outputs\qlora_sft_numina_smoke_1024\numina_train_subset.jsonl
```

---

## 9. Common errors and fixes

### Error: `python` is not recognized

Cause: Python is not installed or not on PATH.

Fix:

```powershell
uv python install 3.11
uv venv .venv --python 3.11 --seed
.\.venv\Scripts\Activate.ps1
```

---

### Error: `py` is not recognized

Cause: Windows Python Launcher is not installed.

Fix: Use `uv` instead of relying on `py`.

---

### Error: `uv` is not recognized

Cause: `uv` was not installed or its install path is not on PATH.

Fix:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv --version
```

---

### Error: `requirements.txt` does not exist

Cause: The repo does not have a `requirements.txt` file.

Fix: Install the minimal dependencies manually:

```powershell
python -m pip install --upgrade pip setuptools wheel numpy
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf scipy tqdm pandas scikit-learn antlr4-python3-runtime==4.11.1
```

---

### Error: vLLM build fails on Windows

Cause: `vllm` does not support native Windows well.

Fix: Do not install `vllm` for native Windows smoke training. Use WSL2, Linux, DSMLP, or Colab if vLLM is required.

---

### Error: `UnicodeDecodeError: 'charmap' codec can't decode...`

Cause: Windows default encoding is not UTF-8.

Fix:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 1024
```

---

## 10. Full clean setup command sequence

Use this when setting up from scratch:

```powershell
cd C:\Users\es-tec\Downloads\151B_SP26_Competition

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

uv python install 3.11
uv venv .venv --python 3.11 --seed
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel numpy
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf scipy tqdm pandas scikit-learn antlr4-python3-runtime==4.11.1

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -X utf8 -c "from trl import SFTTrainer, SFTConfig; print('TRL import OK')"

python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 1024
```

---

## 11. Mental checklist

Before debugging the model, confirm these first:

```text
1. (.venv) is active
2. python --version shows Python 3.11.x
3. torch.cuda.is_available() returns True
4. TRL import works with python -X utf8
5. vLLM is not being installed on native Windows
6. Smoke test runs at max_seq_len 1024 before trying 2048
```

Do not debug QLoRA until the environment passes this checklist.
