# Fine-tune MRPC

## Get Data

```sh
# Download and pre-process MRPC dataset.
python fine_tune/download_mrpc.py
```

## BERT

### BERT Fine-Tune Script

- Train dataset: 3668
  - 1 epoch = 115 iter (batch size =32)
- Dev dataset: 408

```sh
# Fine-tune on MRPC
python run_fine_tune.py \
--experiment teacher_huggingface  \
--model bert                      \
--ptrain_ver bert-base-uncased    \
--task mrpc                       \
--dataset train                   \
--num_class 2                     \
--accum_step 1                    \
--batch_size 32                   \
--beta1 0.9                       \
--beta2 0.999                     \
--ckpt_step 115                   \
--dropout 0.1                     \
--eps 1e-8                        \
--log_step 10                     \
--lr 2e-5                         \
--max_norm 1.0                    \
--max_seq_len 128                 \
--device_id 1                     \
--seed 42                         \
--total_step 575                  \
--warmup_step 172                 \
--weight_decay 0.01
```

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo pkd-even                          \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 575 \
--experiment PKD_hugface_9_5_4_2_soft_26            \
--model bert                       \
--task mrpc                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 100                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 10                     \
--lr 5e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 1150                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step 115               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                 \
--mu 500                          \
--soft_weight 0.7                 \
--hard_weight 0.3
```

### AKD-BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo akd-highway                          \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  575 \
--experiment AKD_hugface_soft_3_26            \
--model bert                       \
--task mrpc                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 100                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 10                     \
--lr 5e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 1150                \
--type_vocab_size 2                \
--warmup_step  115               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--seed 26                          \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on RTE dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment   AKD_hugface_soft_3_26                 \
--model bert                    \
--task mrpc                     \
--dataset train                 \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on RTE dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment   AKD_hugface_soft_3_26                 \
--model bert                    \
--task mrpc                     \
--dataset dev           \
--batch_size 128 \
--device_id 0
```
