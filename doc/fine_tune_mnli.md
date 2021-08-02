# Fine-tune MNLI

## Get Data

```sh
# Download MNLI dataset.
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip

# Move MNLI into data folder.
mv multinli_1.0.zip ./data/fine_tune/mnli.zip

# Extract MNLI from zip.
unzip ./data/fine_tune/mnli.zip -d ./data/fine_tune/mnli

# Format file names.
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl ./data/fine_tune/mnli/dev_matched.jsonl
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl ./data/fine_tune/mnli/dev_mismatched.jsonl
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_train.jsonl ./data/fine_tune/mnli/train.jsonl

# Remove redundant files.
rm -rf ./data/fine_tune_data/mnli/__MACOSX
rm -rf ./data/fine_tune_data/mnli/multinli_1.0
rm ./data/fine_tune/mnli.zip
```

## BERT

### BERT Fine-Tune Script

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
--weight_decay 0.01            \
--amp
```

## BERT-PKD

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment PKD_4layer_soft_65            \
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
--num_hidden_layers 4              \
--total_step 49088                \
--type_vocab_size 2                \
--seed 65                          \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 3                      \
--tdevice_id 3                     \
--softmax_temp 10                  \
--mu 100                           \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## AKD-BERT

### AKD-BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--tmodel bert                      \
--teacher_exp teacher_base                \
--tckpt  36816 \
--experiment AKD_soft_2_26            \
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
--lr 5e-5                          \
--gate_lr 1e-6                     \
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
--seed 26                           \
--device_id 1                      \
--tdevice_id 1                     \
--gate_device_id 1                 \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 500
```

## ALP-KD

```sh
python3.8 run_alp_distil.py \
--alp_exp alp-kd-hidden \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  36816 \
--experiment ALP_KD_4layer_soft_2_26            \
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
--num_hidden_layers 4              \
--total_step 49088                \
--type_vocab_size 2                \
--seed 26                          \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 10                  \
--mu 500                         \
--soft_weight 0.5                  \
--hard_weight 0.5
```

## Evaluation

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment  ALP_KD_4layer_soft_2_46          \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 3
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment  ALP_KD_4layer_soft_2_42          \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 512                \
--device_id 2
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment  ALP_KD_4layer_soft_2_42          \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 512 \
--device_id 2
```

### Plot CLS embedding of last Transformer block

```sh
python3.8 plot_CLS_embedding.py  \
--ckpt 24544                     \
--experiment SCL_1_ce             \
--model bert                     \
--task mnli                      \
--dataset dev_matched            \
--batch_size 128                 \
--device_id 0
```
