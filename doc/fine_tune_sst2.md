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

## BERT

### BERT Fine-Tune Script

- Train dataset: 67349
  - 1 epoch = 2105 iter (batch size = 32)
- Dev dataset: 872

```sh
# Fine-tune on SST-2.
python3.8 run_fine_tune.py     \
--experiment bert_base_teacher                 \
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
--lr 3e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 6315             \
--warmup_step  2105           \
--weight_decay 0.01
```

### BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--teacher_exp bert_base_teacher                \
--tmodel bert                      \
--tckpt  6315 \
--experiment PKD_12            \
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
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 21050                \
--type_vocab_size 2                \
--warmup_step  2105               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--mu 100                           \
--use_hidden_loss                  \
--softmax_temp 5                  \
--soft_weight 0.7                  \
--use_classify_loss
```

### Fine-Tune Distillation with layer wise Contrastive learning

```sh
python3.8 run_layerwise_contrast_distill.py \
--teacher_exp bert_base_teacher                \
--tmodel bert                      \
--tckpt 6315 \
--experiment Contrast_by_sample_13            \
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
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 25260                \
--type_vocab_size 2                \
--warmup_step  8420               \
--weight_decay 0.01                \
--device_id 1                      \
--neg_num 20                    \
--contrast_steps 0           \
--contrast_temp 0.1             \
--softmax_temp 20                \
--soft_label_weight 0.5        \
--contrast_weight 1
```

### Build memory bank

```sh
python3.8 build_membank.py \
--experiment bert_base_teacher \
--model bert \
--task sst2 \
--dataset train \
--ckpt 6315 \
--batch_size 256 \
--device_id 1
```

### Build logits bank

```sh
python3.8 build_logitsbank.py \
--experiment bert_base_teacher \
--model bert \
--task sst2 \
--dataset train \
--ckpt 6315 \
--batch_size 256 \
--device_id 0
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on SST2 dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment Contrast_by_sample_13                 \
--model bert                    \
--task sst2                     \
--dataset train                 \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on SST2 dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment Contrast_by_sample_13                 \
--model bert                    \
--task sst2                     \
--dataset dev           \
--batch_size 512 \
--device_id 1
```
