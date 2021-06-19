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

## BERT

### BERT Fine-Tune Script

- Train dataset: 2490
  - 1 epoch = 77 iter (batch size = 32)
- Dev dataset: 277

```sh
# Fine-tune on SST-2.
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

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo pkd-even                          \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 231 \
--experiment KD_hugface_soft_42_3            \
--model bert                       \
--task rte                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 50                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 10                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 462                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 139               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 1                 \
--mu 0                          \
--soft_weight 0.5
```

### AKD-BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo akd-highway                          \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt  231 \
--experiment debug_2            \
--model bert                       \
--task rte                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 50                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 10                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 770                \
--type_vocab_size 2                \
--warmup_step  231               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--seed 42                          \
--softmax_temp 1                  \
--soft_weight 1                  \
--hard_weight 0                \
--use_classify_loss            \
--mu 100                           \
--use_hidden_loss
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on RTE dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment KD_hugface_soft_42_2                 \
--model bert                    \
--task rte                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

```sh
# Fine-tune evaluation on RTE dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment KD_hugface_soft_42_2                 \
--model bert                    \
--task rte                     \
--dataset dev           \
--batch_size 32 \
--device_id 0
```
