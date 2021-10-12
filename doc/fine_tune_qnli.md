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

## Data Information

- binary classification
- train: 104743
- dev: 5463

## Fine-Tune Script of BERT Teacher

- Batch Size: 16, 32
- Learning Rate: 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4
- Dropout: 0.1
- 1 epochs = 3274 iters
  - under batch size 32

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

## Fine-Tune Distillation Scripts of LAD

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


## Analysis on LAD

### LAD with user defined mapping strategy
> Please refer to Section 5.1 in Analysis.

- `gate_indices`: which Gate block's aggregated knowledge the student will learn.
- `student_indices`: which student layer will be optimized.

> For example, if `gate_indices` = `[12,10]` and `student_indices` = `[6,5]`, the student model's 6 and the 5 layers will learn the knowledge of the 12 and the 10 Gate block.

```sh
python3.8 run_probing_lad.py \
--probing_exp lad_user_defined \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment LAD_user_defined_1_42            \
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
--gate_indices 12,10,8,6,4,2          \
--student_indices 6,5,4,3,2,1             \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 1000
```

### Aggregated hidden states vs. aggregated [CLS] embedding
> Please refer to Section 5.2 in Analysis.

#### Train a student with aggregated [CLS] embedding

```sh
python3.8 run_probing_lad.py \
--probing_exp lad_cls \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment debug_lad_cls            \
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
--device_id 3                      \
--tdevice_id 3                     \
--gate_device_id 3                 \
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

#### Log evaluation loss on Tensorboard
- Rember to set the `softmax_temp`, `soft_weight` and `hard_weight`

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

#### Plot the loss curve (Optional)
- Follow the steps of `plot_loss.ipynb`

### Fine-Tune Distillation Scripts of LAD-NO
> Please refer to Section 5.4 in Analysis.

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

### Partial LAD

> Please refer to Section 5.5 in Analysis.

- `gate_indices`: which Gate block's aggregated knowledge the student will learn.

```sh
python3.8 run_probing_lad.py \
--probing_exp partial_lad \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment partial_LAD_5_42            \
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
--total_step 13096                \
--warmup_step  1309               \
--gate_total_step 13096           \
--gate_warmup_step 1309                \
--weight_decay 0.01                \
--device_id 3                      \
--tdevice_id 3                     \
--gate_device_id 3                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2,4,6,8,10          \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 1000
```

## Analysis on ALP-KD

> Please refer to Section 5.3 in Analysis.
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

## Our implementation of BERT-PKD
### Fine-Tune Distillation Scripts of BERT-PKD
> Please refer to Section 7.1 in Appendix.

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

### PKD with user defined mapping strategy
> Please refer to Table 1.1 in Introduction.

- `teacher_indices`: which teacher layer the student will learn.
- `student_indices`: which student layer will be optimized.

> For example, if `teacher_indices` = `[12,10]` and `student_indices` = `[6,5]`, the student model's 6 and the 5 layers will learn the knowledge of the 12 and the 10 teacher layer.

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

## BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment   partial_LAD_4_42    \
--model bert                    \
--task qnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

```sh
# Fine-tune evaluation on QNLI dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment   partial_LAD_4_42    \
--model bert                    \
--task qnli                     \
--dataset dev           \
--batch_size 512 \
--device_id 0
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
