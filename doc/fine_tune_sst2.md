# Fine-tune SST2

## Get Data

```sh
# Download SST-2 dataset.
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

# Move SST-2 into data folder.
mv SST-2.zip ./data/fine_tune/SST-2.zip

# Extract SST-2 from zip.
unzip ./data/fine_tune/SST-2.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/SST-2.zip
```

## Data Information

- binary classification
- train: 67349
- dev: 872
- test: 1821

## Fine-Tune Script of BERT Teacher

- 1 epochs = 2105 iters
  - batch size = 32

```sh
python3.8 run_fine_tune.py     \
--experiment teacher_huggingface \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task sst2                    \
--dataset train                \
--num_class 2                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 2105               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 500                 \
--lr 2e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 6315             \
--warmup_step  1894           \
--weight_decay 0.01
```

## Fine-Tune Distillation Scripts of LAD

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment LAD_soft_5_42            \
--model bert                       \
--task sst2                        \
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
--log_step 100                     \
--lr 1e-4                          \
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 42100                \
--warmup_step  12630               \
--gate_total_step 42100            \
--gate_warmup_step 12630            \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 1                      \
--tdevice_id 1                    \
--gate_device_id 1                 \
--seed 42                          \
--softmax_temp 20                  \
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
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment LAD_user_defined_5            \
--model bert                       \
--task sst2                        \
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
--total_step 42100                \
--warmup_step  12630               \
--gate_total_step 42100           \
--gate_warmup_step 12630                \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--teacher_indices 12,10,8,6,4          \
--student_indices 6,5,4,3,2              \
--softmax_temp 20                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 1000
```

### Fine-Tune Distillation Scripts of LAD-NO

> Please refer to Section 5.4 in Analysis.

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment LAD_NO_soft_1_42            \
--model bert                       \
--task sst2                        \
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
--gate_lr 1e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 42100                \
--warmup_step  12630               \
--gate_total_step 42100            \
--gate_warmup_step 12630            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 0                      \
--tdevice_id 0                     \
--gate_device_id 0                 \
--seed 42                          \
--softmax_temp 20                  \
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
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment partial_LAD_5_42            \
--model bert                       \
--task sst2                        \
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
--total_step 42100                \
--warmup_step  12630               \
--gate_total_step 42100           \
--gate_warmup_step 12630                \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2,4,6,8,10          \
--student_indices 1,2,3,4,5,6             \
--softmax_temp 20                  \
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
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment ALP_KD_soft_2_42            \
--model bert                       \
--task sst2                        \
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
--total_step 25260                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  2526               \
--weight_decay 0.01                \
--device_id 1                       \
--tdevice_id 1                       \
--softmax_temp 20                  \
--mu 1000                          \
--soft_weight 0.5                  \
--hard_weight 0.5
```

### ALP-KD-v2 Fine-Tune Distillation Scripts

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden-v2 \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  6315 \
--experiment ALP_KD_hidden_v2_soft_1_42            \
--model bert                       \
--task sst2                        \
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
--total_step 25260                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  2526               \
--weight_decay 0.01                \
--device_id 1                       \
--tdevice_id 1                       \
--softmax_temp 20                  \
--mu 1000                          \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## Our implementation of BERT-PKD

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 6315 \
--experiment PKD_hugface_soft_2_42            \
--model bert                       \
--task sst2                        \
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
--total_step 25260                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 2526               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 20                  \
--mu 100                           \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## BERT Fine-Tune Evaluation Scripts

### Fine-tune evaluation on SST2 dataset `train`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  partial_LAD_0_42                 \
--model bert                    \
--task sst2                     \
--dataset train                 \
--batch_size 512                \
--device_id 2
```

### Fine-tune evaluation on SST2 dataset `dev`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  partial_LAD_0_42                 \
--model bert                    \
--task sst2                     \
--dataset dev           \
--batch_size 512 \
--device_id 2
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD_soft_5_42                 \
--model bert                    \
--task sst2                     \
--dataset test                 \
--batch_size 256                \
--ckpt 28000 \
--device_id 1
```
