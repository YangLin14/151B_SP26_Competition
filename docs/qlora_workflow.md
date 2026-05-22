# QLoRA workflow

This workflow is for building a competition-use LoRA adapter without depending
on notebook state.

## 1. Smoke test on the public split

Use this first. It checks that model loading, 4-bit quantization, LoRA attachment,
dataset formatting, training, saving, and metadata writing all work.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_smoke \
  --data-source public \
  --max-train-examples 200 \
  --max-steps 50 \
  --max-seq-len 2048
```

Expected outputs:

```text
outputs/qlora_sft_public_smoke/final_adapter/
outputs/qlora_sft_public_smoke/public_train_split.jsonl
outputs/qlora_sft_public_smoke/public_dev_split.jsonl
outputs/qlora_sft_public_smoke/run_metadata.json
```

Use the saved `public_dev_split.jsonl` as the first adapter validation set. Do
not report train-split accuracy as model quality.

## 2. Larger public-split run

After the smoke test is stable, train on more of the public split.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_public_v1 \
  --data-source public \
  --max-train-examples -1 \
  --max-steps -1 \
  --max-seq-len 2048
```

This is useful for learning the competition output format, but it does not give
strong reasoning supervision because the public data only has final answers.

## 3. CoT SFT run

For a more realistic QLoRA experiment, use a dataset with worked solutions.

```bash
python scripts/qlora_sft_train.py \
  --run-name qlora_sft_numina_5k_v1 \
  --data-source numina \
  --max-train-examples 5000 \
  --max-steps -1 \
  --max-seq-len 4096
```

This requires Hugging Face dataset access on the training machine.

## 4. Evaluate the adapter with vLLM

Fast command-line evaluation:

```bash
python scripts/qlora_vllm_eval.py \
  --adapter-path outputs/qlora_sft_public_smoke/final_adapter \
  --adapter-name qlora_sft_public_smoke \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --output-path results/qlora_sft_public_smoke_eval_50.jsonl
```

Base-model control with the same prompt and scoring:

```bash
python scripts/qlora_vllm_eval.py \
  --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl \
  --n-eval 50 \
  --output-path results/qlora_base_control_eval_50.jsonl
```

The adapter run is meaningful only when compared with the base-model control on
the exact same examples and generation settings.

Notebook evaluation path:

In the vLLM inference notebook, load the adapter:

```python
from vllm.lora.request import LoRARequest

vllm_model = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.75,
    max_model_len=32768,
    max_num_seqs=8,
    enable_prefix_caching=True,
    enable_lora=True,
    max_lora_rank=16,
)

LORA_REQUEST = LoRARequest(
    lora_name="qlora_sft_public_smoke",
    lora_int_id=1,
    lora_path="outputs/qlora_sft_public_smoke/final_adapter",
)

outputs = vllm_model.generate(
    prompts,
    sampling_params=sampling_params_sc,
    lora_request=LORA_REQUEST,
)
```

Compare the adapter against the same prompt, eval subset, max token budget, and
sampling settings as the base model. The adapter is worth using only if it beats
the matching base-model run on held-out questions.
