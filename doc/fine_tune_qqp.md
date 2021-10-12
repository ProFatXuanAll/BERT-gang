# Fine-tune QQP

## Get Data

```sh
# Download QQP dataset.
wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip

# Move QQP into data folder.
mv QQP-clean.zip ./data/fine_tune/QQP-clean.zip

# Extract QQP from zip.
unzip ./data/fine_tune/QQP-clean.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/QQP-clean.zip
```

## Data Information

- binary classification
- train: 363846
- dev: 40430
- test: 390965

## Fine-Tune Script of BERT Teacher

- 1 epoch = 11370 iter
  - batch size = 32

```sh
python run_fine_tune.py   \
--experiment teacher_base   \
--model bert                \
--ptrain_ver bert-base-uncased   \
--task qqp                       \
--dataset train                   \
--num_class 2                      \
--accum_step 1                    \
--batch_size 32                   \
--beta1 0.9                       \
--beta2 0.999                     \
--ckpt_step 11370                 \
--dropout 0.1                     \
--eps 1e-8                        \
--log_step 500                    \
--lr 2e-5                         \
--max_norm 1.0                    \
--max_seq_len 128                 \
--device_id 1                     \
--seed 42                         \
--total_step 34110                \
--warmup_step 10233               \
--weight_decay 0.01
```

## Fine-Tune Distillation Scripts of LAD

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment LAD_soft_1_42            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 3000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--gate_eps 1e-8                    \
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 5e-7                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 56850                \
--warmup_step  5685               \
--gate_total_step 56850            \
--gate_warmup_step 5685            \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--type_vocab_size 2                \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--seed 42                          \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## Analysis on LAD

### Fine-Tune Distillation Scripts of LAD-NO

> Please refer to Section 5.4 in Analysis.

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment LAD_NO_soft_2_42_3rdrun            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--gate_eps 1e-8                    \
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 5e-7                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 56850                \
--warmup_step  5685               \
--gate_total_step 56850            \
--gate_warmup_step 5685            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--seed 42                          \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

### Partial LAD

> Please refer to Section 5.5 in Analysis.

- `gate_indices`: which Gate block's aggregated knowledge the student will learn.

```sh
python3.8 run_probing_lad.py \
--probing_exp partial_lad \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment partial_LAD_5_42            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 3000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 5e-7                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 56850                \
--warmup_step  5685               \
--gate_total_step 56850           \
--gate_warmup_step 5685            \
--weight_decay 0.01                \
--device_id 3                      \
--tdevice_id 3                     \
--gate_device_id 3                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2,4,6,8,10                 \
--student_indices 1,2,3,4,5,6      \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## Analysis on ALP-KD

> Please refer to Section 5.3 in Analysis.

### ALP-KD Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment ALP_KD_hidden_soft_2_42            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 45480                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 10                  \
--mu 500                         \
--soft_weight 0.5                  \
--hard_weight 0.5
```

### ALP-KD-v2 Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment ALP_KD_hidden_v2_soft_2_1_65            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 45480                \
--type_vocab_size 2                \
--seed 65                          \
--warmup_step  4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 500                         \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## Our implementation of BERT-PKD

## Fine-Tune Distillation Scripts of BERT-PKD

> Please refer to Section 7.1 in Appendix.

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt 34110 \
--experiment PKD_soft_1_65            \
--model bert                       \
--task qqp                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 3000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 45480                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 100                           \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## BERT Fine-Tune Evaluation Scripts

### Fine-tune evaluation on QQP dataset `train`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment AKD_soft_9_42                     \
--model bert                    \
--task qqp                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

### Fine-tune evaluation on QQP dataset `dev`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment partial_LAD_0_42                     \
--model bert                    \
--task qqp                     \
--dataset dev           \
--batch_size 512 \
--device_id 3
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD_soft_1_42                 \
--model bert                    \
--task qqp                     \
--dataset test                 \
--batch_size 256                \
--ckpt 56850 \
--device_id 1
```
