# Fine-tune MNLI

## Get Data

```sh
# Download MNLI dataset.
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip

# Move MNLI into data folder.
mv MNLI.zip ./data/fine_tune/MNLI.zip

# Extract MNLI from zip.
unzip ./data/fine_tune/MNLI.zip -d ./data/fine_tune/

# Remove redundant file.
rm ./data/fine_tune/MNLI.zip
```

## Data Information

- 3-class classification
- train: 392702
- dev_matched: 9815
- dev_mismatched: 9832
- test_matched: 9796
- test_mismatched: 9847


## Fine-Tune Script of BERT Teacher

- 1 epoch = 12272 iters
  - batch size = 32


```sh
# Fine-tune on MNLI.
python3.8 run_fine_tune.py     \
--experiment teacher_base                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task mnli                    \
--dataset train                \
--num_class 3                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 3000               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 500                 \
--lr 3e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 36816             \
--warmup_step  10000           \
--weight_decay 0.01
```

## Fine-Tune Distillation Scripts of LAD

```sh
python3.8 run_lad_distil.py \
--tmodel bert                      \
--teacher_exp teacher_base                \
--tckpt  36816 \
--experiment LAD_soft_4_6            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 200                     \
--lr 1e-4                          \
--gate_lr 5e-7                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 49088                \
--warmup_step  4908               \
--gate_total_step 49088            \
--gate_warmup_step 4908            \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--seed 42                           \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 500
```

## Analysis on LAD

### Fine-Tune Distillation Scripts of LAD-NO

> Please refer to Section 5.4 in Analysis.

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment LAD_NO_soft_1_42            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--gate_eps 1e-8                    \
--log_step 200                     \
--lr 5e-5                          \
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 49088                \
--warmup_step  4908               \
--gate_total_step 49088            \
--gate_warmup_step 4908            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 500
```

### Partial LAD

```sh
python3.8 run_probing_lad.py \
--probing_exp partial_lad \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment partial_LAD_5_42            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 200                     \
--lr 1e-4                          \
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 49088                \
--warmup_step  4908               \
--gate_total_step 49088           \
--gate_warmup_step 4908            \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2,4,6,8,10                 \
--student_indices 1,2,3,4,5,6      \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 500
```

## Analysis on ALP-KD

> Please refer to Section 5.3 in Analysis.

### ALP-KD Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment ALP_KD_hidden_soft_2_42            \
--model bert                       \
--task mnli                        \
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
--total_step 49088                \
--type_vocab_size 2                \
--seed 65                          \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
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
--tckpt  36816 \
--experiment ALP_KD_hidden_v2_soft_3_65            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 200                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 49088                \
--type_vocab_size 2                \
--seed 65                          \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 10                  \
--mu 100                         \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## Our implementation of BERT-PKD

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

> Please refer to Section 7.1 in Appendix.

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment PKD_even_42            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 200                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 49088                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 10                  \
--mu 500                           \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## BERT Fine-Tune Evaluation Scripts

### Fine-tune evaluation on MNLI dataset `train`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  LAD_soft_4_6          \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

### Fine-tune evaluation on MNLI dataset `dev_matched`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  LAD_soft_4_6          \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 512                \
--device_id 0
```

### Fine-tune evaluation on MNLI dataset `dev_mismatched`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  LAD_soft_4_6          \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 512 \
--device_id 0
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD_soft_4_6                 \
--model bert                    \
--task mnli                     \
--dataset test_mismatched                 \
--batch_size 256                \
--ckpt 38000 \
--device_id 0
```
