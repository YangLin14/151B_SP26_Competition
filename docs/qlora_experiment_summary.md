# QLoRA experiment summary

This file records small, curated results so the repository does not need to
track generated adapters, checkpoints, or raw model outputs.

## qlora_sft_public_smoke

Run purpose: engineering smoke test for the QLoRA training/evaluation pipeline.

Training setup:

```text
model: Qwen/Qwen3-4B-Thinking-2507
data_source: public
max_train_examples: 200
max_steps: 50
max_seq_len: 1024 on Windows
```

Evaluation setup:

```text
backend: Transformers on native Windows
held-out split: outputs/qlora_sft_public_smoke/public_dev_split.jsonl
n_eval: 10
max_input_length: 1024
max_new_tokens: 1024
```

Results:

| Run | Correct | Notes |
|---|---:|---|
| Base control | 4 / 10 | Often generated hundreds to 1024 tokens. |
| QLoRA public smoke adapter | 3 / 10 | Generated much shorter answers, around 19-53 tokens. |

Conclusion:

```text
The public-smoke adapter proves the QLoRA pipeline works, but it should not be
used as the competition model. It learned short-answer behavior and was slightly
worse than the base control on the 10-example held-out smoke eval.
```

Next step:

```text
Use NuminaMath CoT streaming smoke training before scaling QLoRA, because the
public dataset only provides final answers and does not supervise reasoning.
```
