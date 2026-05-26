Pipeline 是什麼
整體 QLoRA pipeline 是：

setup env
-> smoke train
-> verify adapter files
-> base control eval
-> adapter eval
-> compare scores
-> scale training
-> final eval / decide whether to use adapter

1. Setup env

這一步只是確保 Python、CUDA、PyTorch、TRL、PEFT、bitsandbytes 都能用。

Windows 用：

python -c "import torch; print(torch.cuda.is_available())"
python -X utf8 -c "from trl import SFTTrainer, SFTConfig; print('TRL import OK')"

如果這裡不過，後面都不用跑。

2. Smoke train

例如：

python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_smoke --data-source public --max-train-examples 200 --max-steps
50 --max-seq-len 1024

這不是為了拿高分。它是工程測試，確認：

model can load
4-bit quantization works
LoRA adapter attaches
dataset formatting works
training loop runs
adapter can be saved
metadata is written

輸出在：

outputs/qlora_sft_public_smoke/final_adapter/
outputs/qlora_sft_public_smoke/run_metadata.json
outputs/qlora_sft_public_smoke/public_train_split.jsonl
outputs/qlora_sft_public_smoke/public_dev_split.jsonl

3. Public train/dev split

我們不能把 public.jsonl 全部拿去 train，然後又說 public accuracy 很高。那是 train-on-test。

所以 script 會把 public data 切成：

80% train
20% held-out dev

Training 只看 train split。Evaluation 要看 dev split。

4. Base control eval

這一步評估「沒有 LoRA adapter 的原始模型」在同一批 dev 題上的分數。

例如 Windows：

python -X utf8 scripts/qlora_transformers_eval.py --data-path outputs/qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-
input-length 1024 --max-new-tokens 1024 --output-path results/qlora_base_control_transformers_eval_50.jsonl

這是 control group。沒有它，你不知道 QLoRA 有沒有真的變好。

5. Adapter eval

這一步評估「掛上 QLoRA adapter 的模型」。

python -X utf8 scripts/qlora_transformers_eval.py --adapter-path outputs/qlora_sft_public_smoke/final_adapter --data-path outputs/
qlora_sft_public_smoke/public_dev_split.jsonl --n-eval 50 --max-input-length 1024 --max-new-tokens 1024 --output-path results/
qlora_sft_public_smoke_transformers_eval_50.jsonl

它必須跟 base control 用同一個：

data-path
n-eval
max-input-length
max-new-tokens
temperature
top-p
top-k

不然分數不能公平比較。

6. Compare

eval script 會印：

MCQ : x / y
Free-form : x / y
Overall : x / y

判斷：

adapter overall > base overall -> 可以考慮繼續放大
adapter overall <= base overall -> 先不要放大，改資料/參數/target

7. Real training

如果 smoke 成功，才跑 real training。

Public-format adapter：

python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_public_v1_1024 --data-source public --max-train-examples -1 --max-steps
-1 --max-seq-len 1024

這會學 competition answer format，但因為 public data 只有 final answer，對 reasoning 幫助有限。

NuminaMath CoT adapter：

python -X utf8 scripts/qlora_sft_train.py --run-name qlora_sft_numina_5k_1024 --data-source numina --max-train-examples 5000 --max-
steps -1 --max-seq-len 1024

這比較像真正 SFT，因為 Numina 有 worked solutions，可以訓練 reasoning trace。

8. Final decision

最後你會有幾組結果：

base control
public QLoRA adapter
Numina CoT QLoRA adapter
maybe prompt_v2 from teammate
maybe prompt_v2 + adapter

最終 competition 要用的應該是：

best prompt + best model/adaptor + best decoding settings

但每次只改一個東西，不然你不知道 improvement 來自哪裡。

我建議你現在的順序

1. 確認 qlora_sft_public_smoke adapter 有產出。
2. 跑 base control eval 10 題。
3. 跑 adapter eval 10 題。
4. 如果都能跑，再跑 50 題。
5. 如果 adapter 沒贏，不要急著 full train public；先跑 Numina smoke。
6. 如果 Numina smoke 能跑，再跑 Numina 5k。
