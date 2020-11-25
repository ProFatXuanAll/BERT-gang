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
--experiment test                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task mnli                    \
--dataset train                \
--num_class 3                  \
--accum_step 8                 \
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
--device_id 0                     \
--seed 42                      \
--total_step 36816             \
--warmup_step  10000           \
--weight_decay 0.01            \
--amp
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment test                  \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment test                  \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment test                  \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 128
```

### BERT Fine-Tune Experiment Results

- Shared configuration

|beta1|beta2|eps|max_norm|weight_decay|
|-|-|-|-|-|
|0.9|0.999|1e-8|1.0|0.01|

- Individual configuration

|ex|train acc|train acc ckpt|dev-m acc|dev-m acc ckpt|dev-mm acc|dev-mm acc ckpt|accum step|batch|ckpt step|dropout|encoder|log step|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0.983002|36816|0.867753|36000|0.862083|36000|8|32|1000|0.1|bert-large-uncased|500|3e-5|128|42|36816|10000|
|2|0.964217|36816|0.846052|24000|0.850488|36816|8|32|1000|0.1|bert-base-uncased|500|3e-5|128|42|36816|10000|

### BERT Fine-Tune Distillation Scripts with Multi-GPU

#### Use logits loss + hidden states loss + attention loss

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--teacher_exp test                \
--tmodel bert                      \
--tckpt  36816 \
--experiment distill_2_6_2             \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 100000                \
--type_vocab_size 2                \
--warmup_step  30000               \
--weight_decay 0.01                \
--device_id 1                      \
--use_logits_loss                  \
--use_hidden_loss                  \
--use_attn_loss                    \
--amp
```

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment distill_2_6_2          \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 0
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_2_6_2          \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 512
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_PKD_NoSoftmaxTEMP2          \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 512
```

## ALBERT

### ALBERT Fine-Tune Script

```sh
# Fine-tune on MNLI.
python3.8 run_fine_tune.py  \
--experiment 1              \
--model albert              \
--ptrain_ver albert-base-v2 \
--task mnli                 \
--dataset train             \
--num_class 3               \
--accum_step 8              \
--batch_size 128            \
--beta1 0.9                 \
--beta2 0.999               \
--ckpt_step 500             \
--dropout 0.1               \
--eps 1e-8                  \
--log_step 250              \
--lr 3e-5                   \
--max_norm 1.0              \
--max_seq_len 512           \
--num_gpu 1                 \
--seed 42                   \
--total_step 10000          \
--warmup_step  1000         \
--weight_decay 0.01         \
--amp
```

### ALBERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model albert                  \
--task mnli                     \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model albert                  \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model albert                  \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 128
```

### ALBERT Fine-Tune Experiment Results

- Shared configuration

|beta1|beta2|eps|max_norm|weight_decay|
|-|-|-|-|-|
|0.9|0.999|1e-8|1.0|0.01|

- Individual configuration

|ex|train acc|train acc ckpt|dev-m acc|dev-m acc ckpt|dev-mm acc|dev-mm acc ckpt|accum step|batch|ckpt step|dropout|encoder|log step|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0|0|0.848395|5500|0.851810|6500|8|128|500|0.1|albert-base-v2|250|3e-5|192|42|10000|1000|

### ALBERT Fine-Tune Logits Generation Scripts

```sh
# Generate MNLI logits.
python3.8 run_fine_tune_gen_logits.py \
--experiment 1                        \
--model albert                        \
--task mnli                           \
--dataset train                       \
--ckpt 0                              \
--batch_size 128
```

### ALBERT Fine-Tune Distillation Scripts

```sh
# Fine-tune distillation on MNLI.
python3.8 run_fine_tune_distill.py \
--experiment distill_1             \
--model albert                     \
--task mnli                        \
--dataset 1_albert_mnli            \
--num_class 3                      \
--accum_step 8                     \
--batch_size 128                   \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_emb 128                        \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--max_seq_len 512                  \
--num_attention_heads 16           \
--num_gpu 1                        \
--num_hidden_layers 6              \
--seed 42                          \
--total_step 100000                \
--type_vocab_size 2                \
--warmup_step  10000               \
--weight_decay 0.01         \
--amp
```

### ALBERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model albert                  \
--task mnli                     \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model albert                  \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model albert                  \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 128
```
