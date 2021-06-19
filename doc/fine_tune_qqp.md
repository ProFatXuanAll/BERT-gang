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

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo pkd-even                          \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt 34110 \
--experiment PKD_even_soft_1_26            \
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
--seed 26                          \
--warmup_step 4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 20                  \
--mu 100                           \
--soft_weight 0.7
```

### AKD-BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo akd-highway                          \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  34110 \
--experiment AKD_soft_1_26            \
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
--warmup_step  4548               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--seed 26                          \
--softmax_temp 20                  \
--soft_weight 0.7                  \
--hard_weight 0.3                \
--mu 100                           \
--use_hidden_loss                  \
--use_classify_loss
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QQP dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment PKD_even_soft_1_46                 \
--model bert                    \
--task qqp                     \
--dataset train                 \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on QQP dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment PKD_even_soft_1_26                 \
--model bert                    \
--task qqp                     \
--dataset dev           \
--batch_size 512 \
--device_id 1
```
