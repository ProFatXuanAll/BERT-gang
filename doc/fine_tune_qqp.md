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

## BERT Fine-Tune

### BERT Fine-Tune Script

- Train dataset: 363846
  - 1 epoch = 11370 iter (batch size = 32)
- Dev dataset: 40430

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

## BERT-PKD

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt 34110 \
--experiment PKD_4layer_soft_1_26            \
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
--num_hidden_layers 4              \
--total_step 45480                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step 4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 100                           \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## LAD

### LAD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment AKD_4layer_soft_6_42            \
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
--num_hidden_layers 4              \
--total_step 56850                \
--warmup_step  5685               \
--gate_total_step 56850            \
--gate_warmup_step 5685            \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--type_vocab_size 2                \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--seed 42                          \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

## Probing Experiments

### Partial LAD

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

## ALP-KD

### ALP-KD training scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment ALP_KD_hidden_v2_4layer_soft_2_1_26            \
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
--num_hidden_layers 4              \
--total_step 45480                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 500                         \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## Evaluation

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QQP dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment AKD_soft_9_42                     \
--model bert                    \
--task qqp                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

```sh
# Fine-tune evaluation on QQP dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment partial_LAD_5_42                     \
--model bert                    \
--task qqp                     \
--dataset dev           \
--batch_size 512 \
--device_id 0
```

### Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  PKD_soft_1_65                 \
--model bert                    \
--task qqp                     \
--dataset test                 \
--batch_size 256                \
--ckpt 45480 \
--device_id 1
```
