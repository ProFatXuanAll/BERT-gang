# Fine-tune QNLI

## Get Data

```sh
# Download QNLI dataset.
wget https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip

# Move QNLI into data folder.
mv QNLIv2.zip ./data/fine_tune/qnli.zip

# Extract QNLI from zip.
unzip ./data/fine_tune/qnli.zip -d ./data/fine_tune

# Remove redundant files.
rm ./data/fine_tune/qnli.zip
```

- binary classification
- train: 104743
- dev: 5463

## BERT Teacher

- Batch Size: 16, 32
- Learning Rate: 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4
- Dropout: 0.1
- 1 epochs = 3274 iters
  - under batch size 32

### BERT Fine-Tune Script

```sh
python3.8 run_fine_tune.py     \
--experiment teacher_base       \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task qnli                    \
--dataset train                \
--num_class 2                  \
--accum_step 2                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 1000               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 500                 \
--lr 3e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 0                     \
--seed 42                      \
--total_step 9822             \
--warmup_step 3274            \
--weight_decay 0.01
```

## BERT-PKD Fine-Tune Distillation Scripts

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment PKD_soft_1_46            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--type_vocab_size 2                \
--seed 46                          \
--warmup_step  1309               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 10                  \
--mu 100                           \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## LAD Fine-Tune Distillation Scripts

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment LAD_soft_2_42            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 1e-7                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 13096                \
--warmup_step  1309               \
--gate_total_step 13096           \
--gate_warmup_step 1309                \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 1000
```

## LAD-NO

### LAD-NO Fine-Tune Distillation Scripts

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment LAD_NO_soft_1_42            \
--model bert                       \
--task qnli                        \
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
--lr 1e-4                          \
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--warmup_step  1309               \
--gate_total_step 13096            \
--gate_warmup_step 1309            \
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
--mu 1000
```

## Probing Experiments

### PKD with user defined mapping strategy

```sh
python3.8 run_probing.py \
--probing_exp pkd_hidden_user_defined \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment PKD_Hidden_user_defined_10_42 \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 32740                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  3274               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--teacher_indices 12,10,8,6,4,2           \
--student_indices 6,5,4,3,2,1              \
--softmax_temp 10                  \
--mu 100                           \
--soft_weight 0.5                  \
--hard_weight 0.5
```

### LAD with user defined mapping strategy

```sh
python3.8 run_probing_lad.py \
--probing_exp lad_user_defined \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment AKD_CLS_user_defined_1_42            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 32740                \
--warmup_step  3274               \
--gate_total_step 32740           \
--gate_warmup_step 3274                \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--teacher_indices 12,10,8,6,4,2          \
--student_indices 6,5,4,3,2,1              \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 1000
```

## ALP-KD

### ALP-KD Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment ALP_KD_hidden_soft_5_2_42            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 1e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--type_vocab_size 2                \
--seed 42                         \
--warmup_step  1309               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 10                  \
--mu 1000                          \
--soft_weight 0.5                  \
--hard_weight 0.5
```

### ALP-KD-v2 Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment ALP_KD_hidden_v2_soft_3_2_42            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 3e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--type_vocab_size 2                \
--seed 42                         \
--warmup_step  1309               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 10                  \
--mu 5000                          \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment   LAD_NO_soft_1_42    \
--model bert                    \
--task qnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

```sh
# Fine-tune evaluation on QNLI dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment   LAD_NO_soft_1_42    \
--model bert                    \
--task qnli                     \
--dataset dev           \
--batch_size 512 \
--device_id 0
```

## Plot CLS embedding of last Transformer block

```sh
python3.8 plot_CLS_embedding.py  \
--ckpt 11000                     \
--experiment LAD_NO_soft_1_42             \
--model bert                     \
--task qnli                      \
--dataset dev            \
--batch_size 256                 \
--device_id 1
```

## Evaluate validation loss

```sh
python3.8 eval_dev_loss.py \
--experiment AKD_CLS_user_defined_1_42 \
--texperiment LAD_NO_soft_1_42 \
--model bert \
--tmodel bert \
--tckpt 9822 \
--task qnli \
--dataset dev \
--batch_size 32 \
--log_step 17 \
--tdevice_id 1 \
--device_id 1 \
--softmax_temp 10 \
--soft_weight 0.5 \
--hard_weight 0.5
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD_NO_soft_1_42                 \
--model bert                    \
--task qnli                     \
--dataset test                 \
--batch_size 256                \
--ckpt 11000 \
--device_id 0
```
