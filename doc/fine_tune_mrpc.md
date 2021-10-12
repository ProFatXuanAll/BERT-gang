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

## BERT-PKD

### BERT-PKD Fine-Tune Distillation Scripts

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 575 \
--experiment partial_LAD_0_42            \
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
--lr 7e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 2300                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 690               \
--weight_decay 0.01                \
--device_id 2                      \
--tdevice_id 2                     \
--softmax_temp 10                 \
--mu 500                          \
--soft_weight 0.7                 \
--hard_weight 0.3
```

## LAD

### LAD Fine-Tune Distillation Scripts

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  575 \
--experiment LAD6_soft_2_3_26            \
--model bert                       \
--task mrpc                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 100                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--gate_eps 1e-8                    \
--log_step 10                     \
--lr 7e-4                          \
--gate_lr 5e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--type_vocab_size 2                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 2300                \
--warmup_step  690               \
--gate_total_step 2300            \
--gate_warmup_step 690             \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 2                      \
--tdevice_id 2                     \
--gate_device_id 2                 \
--seed 26                          \
--softmax_temp 10                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## Probing Experiments

### Partial LAD

```sh
python3.8 run_probing_lad.py \
--probing_exp partial_lad \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  575 \
--experiment partial_LAD_5_42            \
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
--lr 7e-4                          \
--gate_lr 5e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 2300                \
--warmup_step  690               \
--gate_total_step 2300           \
--gate_warmup_step 690            \
--weight_decay 0.01                \
--device_id 2                      \
--tdevice_id 2                     \
--gate_device_id 2                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2,4,6,8,10                 \
--student_indices 1,2,3,4,5,6      \
--softmax_temp 10                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## LAD-NO

### LAD-NO Fine-Tune Distillation Scripts

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  575 \
--experiment LAD_NO_soft_1_42            \
--model bert                       \
--task mrpc                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 100                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--gate_eps 1e-8                    \
--log_step 10                     \
--lr 5e-4                          \
--gate_lr 5e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 2300                \
--warmup_step  690               \
--gate_total_step 2300            \
--gate_warmup_step 690            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## ALP-KD

### ALP-KD Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  575 \
--experiment ALP_KD_hidden_4layer_soft_2_26            \
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
--num_hidden_layers 4              \
--total_step 1150                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  115               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 1000                          \
--soft_weight 0.7                  \
--hard_weight 0.3
```

### ALP-KD-v2 Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  575 \
--experiment ALP_KD_hidden_v2_4layer_soft_2_26            \
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
--num_hidden_layers 4              \
--total_step 1150                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  115               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 1000                          \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on RTE dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment  partial_LAD_4_42  \
--model bert                    \
--task mrpc                     \
--dataset train                 \
--batch_size 256                \
--device_id 3
```

```sh
# Fine-tune evaluation on RTE dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment  partial_LAD_4_42  \
--model bert                    \
--task mrpc                     \
--dataset dev           \
--batch_size 128 \
--device_id 3
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD6_soft_2_26                 \
--model bert                    \
--task mrpc                     \
--dataset test                 \
--batch_size 256                \
--ckpt 1900 \
--device_id 0
```
