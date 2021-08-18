# Fine-tune CoLA

## Get Data

```sh
# Download CoLA dataset.
wget https://dl.fbaipublicfiles.com/glue/data/CoLA.zip

# Move CoLA to data folder.
mv CoLA.zip ./data/fine_tune/CoLA.zip

# Extract CoLA from zip.
unzip ./data/fine_tune/CoLA.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/CoLA.zip
```

## BERT

### BERT Fine-Tune Script

- Train dataset: 8551
  - 1 epoch = 268 (batch size = 32)
- Dev dataset: 1043

```sh
# Fine-tune CoLA
python3.8 run_fine_tune.py     \
--experiment teacher_huggingface                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task cola                    \
--dataset train                \
--num_class 2                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 268               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 10                 \
--lr 2e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 804             \
--warmup_step  80           \
--weight_decay 0.01
```

## BERT-PKD

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_pkd_distill.py \
--teacher_exp teacher_huggingface                \
--tmodel bert                      \
--tckpt 231 \
--experiment PKD_soft_1_42            \
--model bert                       \
--task cola                        \
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
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 1340                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step 134               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--softmax_temp 5                 \
--mu 500                          \
--soft_weight 0.7                 \
--hard_weight 0.3
```
