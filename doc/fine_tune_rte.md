# Fine-tune RTE

## Get Data

```sh
# Download RTE dataset.
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip

# Move RTE into data folder.
mv RTE.zip ./data/fine_tune/RTE.zip

# Extract RTE from zip
unzip ./data/fine_tune/RTE.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/RTE.zip
```

## Data Information

- binary classification
- train: 2490
- dev: 277
- test: 3000

## Fine-Tune Script of BERT Teacher

- 1 epoch = 77 iter
  - batch size = 32

```sh
python3.8 run_fine_tune.py     \
--experiment teacher_huggingface                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task rte                    \
--dataset train                \
--num_class 2                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 77               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 10                 \
--lr 2e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 231             \
--warmup_step  69           \
--weight_decay 0.01
```

## Fine-Tune Distillation Scripts of LAD

```sh
python3.8 run_lad_distil.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  231 \
--experiment LAD_soft_3_26            \
--model bert                       \
--task rte                        \
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
--lr 3e-4                          \
--gate_lr 3e-5                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 4              \
--total_step 1155                \
--warmup_step  346               \
--gate_total_step 1155             \
--gate_warmup_step 346            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
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
--tckpt  231 \
--experiment LAD_user_defined_6            \
--model bert                       \
--task rte                        \
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
--lr 3e-4                          \
--gate_lr 3e-6                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 1540                \
--warmup_step  462               \
--gate_total_step 1540           \
--gate_warmup_step 462                \
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
--softmax_temp 5                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 500
```

### Fine-Tune Distillation Scripts of LAD-NO

> Please refer to Section 5.4 in Analysis.

```sh
python3.8 run_lad_no_distil.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  231 \
--experiment LAD_NO_soft_5_42            \
--model bert                       \
--task rte                        \
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
--lr 9e-4                          \
--gate_lr 3e-5                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                 \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 1540                \
--warmup_step  462               \
--gate_total_step 1540            \
--gate_warmup_step 462            \
--type_vocab_size 2                \
--weight_decay 0.01                \
--gate_weight_decay 0.01           \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--seed 42                          \
--softmax_temp 5                  \
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
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  231 \
--experiment partial_LAD_1_42            \
--model bert                       \
--task rte                        \
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
--lr 3e-4                          \
--gate_lr 3e-5                     \
--max_norm 1.0                     \
--gate_max_norm 1.0                \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--type_vocab_size 2                \
--total_step 1155                \
--warmup_step  346               \
--gate_total_step 1155           \
--gate_warmup_step 346            \
--weight_decay 0.01                \
--device_id 2                      \
--tdevice_id 2                     \
--gate_device_id 2                 \
--gate_beta1 0.9                   \
--gate_beta2 0.999                 \
--gate_eps 1e-8                    \
--gate_weight_decay 0.01           \
--seed 42                          \
--gate_indices 2                 \
--student_indices 1,2,3,4,5,6      \
--softmax_temp 5                  \
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
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  231 \
--experiment ALP_KD_soft_4_26            \
--model bert                       \
--task rte                        \
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
--total_step 1155                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  346               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 5                  \
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
--tckpt  231 \
--experiment ALP_KD_hidden_v2_soft_1_26            \
--model bert                       \
--task rte                        \
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
--lr 3e-4                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 1155                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  346               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 5                  \
--mu 500                          \
--soft_weight 0.7                  \
--hard_weight 0.3
```

## Our implementation of BERT-PKD

### Fine-Tune Distillation Scripts of BERT-PKD

> Please refer to Section 7.1 in Appendix.

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 231 \
--experiment PKD_hugface_soft_6_2_42            \
--model bert                       \
--task rte                        \
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
--total_step 1540                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 462               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 5                 \
--mu 500                          \
--soft_weight 0.7                 \
--hard_weight 0.3
```

## BERT Fine-Tune Evaluation Scripts

### Fine-tune evaluation on RTE dataset `train`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  LAD_soft_3_26                 \
--model bert                    \
--task rte                     \
--dataset train                 \
--batch_size 256                \
--device_id 1
```

### Fine-tune evaluation on RTE dataset `dev`.

```sh
python3.8 run_fine_tune_eval.py \
--experiment  LAD_soft_3_26                 \
--model bert                    \
--task rte                     \
--dataset dev           \
--batch_size 256 \
--device_id 1
```

## Generate prediction result

```sh
python3.8 generate_test_prediction.py \
--experiment  LAD_soft_3_26                 \
--model bert                    \
--task rte                     \
--dataset test                 \
--batch_size 256                \
--ckpt 1000 \
--device_id 0
```
