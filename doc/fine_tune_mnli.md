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
--experiment 1                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task mnli                    \
--dataset train                \
--num_class 3                  \
--accum_step 1                 \
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
--num_gpu 1                    \
--seed 42                      \
--total_step 36816             \
--warmup_step  10000           \
--weight_decay 0.01
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
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
|1|0|0|0.846256|95000|0.848047|71000|1|32|1000|0.1|bert-base-uncased|500|3e-5|128|42|100000|10000|

### BERT Fine-Tune Logits Generation Scripts

```sh
# Generate MNLI logits.
python3.8 run_fine_tune_gen_logits.py \
--experiment 1                        \
--model bert                          \
--task mnli                           \
--dataset train                       \
--ckpt 0                              \
--batch_size 128
```

### BERT Fine-Tune Distillation Scripts

```sh
# Fine-tune distillation on MNLI.
python3.8 run_fine_tune_distill.py \
--experiment distill_1             \
--model bert                       \
--task mnli                        \
--dataset 1_bert_mnli              \
--num_class 3                      \
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
--max_seq_len 128                  \
--num_attention_heads 16           \
--num_gpu 1                        \
--num_hidden_layers 6              \
--seed 42                          \
--total_step 100000                \
--type_vocab_size 2                \
--warmup_step  10000               \
--weight_decay 0.01
```

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 128
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
--weight_decay 0.01
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
--weight_decay 0.01
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
