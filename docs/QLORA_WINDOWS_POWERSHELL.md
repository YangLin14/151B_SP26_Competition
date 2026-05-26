# QLoRA Windows PowerShell Pipeline

Use this on native Windows. Do not install or run `vllm` here.

Project root:

```powershell
cd C:\Users\es-tec\Downloads\151B_SP26_Competition
```

## Setup

```powershell
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
```

Do not run:

```powershell
python -m pip install vllm
python scripts/qlora_vllm_eval.py --help
```

## Public Smoke Training

Start conservative on RTX 3060/3060 Ti:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 1024
```

If stable, try 2048:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke_2048 --data-source public --max-train-examples 200 --max-steps 50 --max-seq-len 2048
```

Check output:

```powershell
Get-ChildItem outputs\qlora_sft_public_smoke
Get-ChildItem outputs\qlora_sft_public_smoke\final_adapter
Get-Content outputs\qlora_sft_public_smoke\run_metadata.json
```

## Windows Evaluation With Transformers

Base control, first 10 examples:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 10 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_base_control_transformers_eval_10.jsonl
```

Adapter eval, same examples/settings:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_public_smoke/final_adapter --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 10 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_sft_public_smoke_transformers_eval_10.jsonl
```

If 10 examples work, run 50:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_base_control_transformers_eval_50.jsonl
```

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_public_smoke/final_adapter --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_sft_public_smoke_transformers_eval_50.jsonl
```

If generation finished but scoring failed, reuse the raw generations:

```powershell
python -m pip install antlr4-python3-runtime==4.11.1
python -X utf8 scripts/qlora_score_raw.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --raw-path results/qlora_base_control_transformers_eval_10.raw.jsonl --output-path results/qlora_base_control_transformers_eval_10.jsonl
```

## Public Real Training

This trains on the public 80% train split and keeps 20% held out.

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_v1_1024 --data-source public --max-train-examples -1 --max-steps -1 --max-seq-len 1024
```

Only if VRAM is stable:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_v1_2048 --data-source public --max-train-examples -1 --max-steps -1 --max-seq-len 2048
```

Evaluate:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --data-path outputs/qlora_sft_public_v1_1024/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_public_v1_base_control_transformers_eval_50.jsonl
```

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_public_v1_1024/final_adapter --data-path outputs/qlora_sft_public_v1_1024/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_public_v1_adapter_transformers_eval_50.jsonl
```

## NuminaMath CoT Training

The script streams NuminaMath by default and saves the selected subset under the run output directory.

Smoke:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_numina_smoke_1024 --data-source numina --max-train-examples 500 --max-steps 50 --max-seq-len 1024 --numina-shuffle-buffer 10000
```

Larger run:

```powershell
python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_numina_5k_1024 --data-source numina --max-train-examples 5000 --max-steps -1 --max-seq-len 1024 --numina-shuffle-buffer 10000
```

Evaluate:

```powershell
python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_numina_5k_1024/final_adapter --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/qlora_numina_5k_transformers_eval_50.jsonl
```

## Read Results

```powershell
Get-ChildItem results
Get-Content results\qlora_sft_public_smoke_transformers_eval_50.jsonl -TotalCount 2
Get-Content results\qlora_sft_public_smoke_transformers_eval_50.metadata.json
```

Decision rule:

```text
Keep scaling QLoRA only if adapter eval > base control eval on the same held-out examples.
Do not use train-split accuracy as evidence.
Do not compare Windows Transformers numbers directly against vLLM numbers unless settings match closely.
```
