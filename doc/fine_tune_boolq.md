# Fine-tune BoolQ

- [official github link](https://github.com/google-research-datasets/boolean-questions)

## Get Data

```sh
# Download BoolQ dataset.
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip

# Move BoolQ into data folder.
mv BoolQ.zip ./data/fine_tune/BoolQ.zip

# Extract BoolQ from zip.
unzip ./data/fine_tune/BoolQ.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/BoolQ.zip
```

## BERT

### BERT Fine-Tune Script

```sh
# Fine-tune on BoolQ.
python3.8 run_fine_tune.py     \
--experiment 1                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task boolq                   \
--dataset train                \
--num_class 2                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 1000               \
--dropout 0.1                  \
--eps 1e-8                     \
--lr 1e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--num_gpu 1                    \
--seed 42                      \
--total_step 100000            \
--warmup_step  10000           \
--weight_decay 0.01
```

### BERT Fine-tune Evaluation Scripts

```sh
# Fine-tune evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model bert                    \
--task boolq                    \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model bert                    \
--task boolq                    \
--dataset val                   \
--batch_size 128
```

### BERT Fine-tune Experiment Results

- Shared configuration

|beta1|beta2|eps|max_norm|weight_decay|
|-|-|-|-|-|
|0.9|0.999|1e-8|1.0|0.01|

- Individual configuration

|ex|train acc|train acc ckpt|dev-m acc|dev-m acc ckpt|dev-mm acc|dev-mm acc ckpt|accum step|batch|ckpt step|dropout|encoder|log step|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0|0|0|0|0|0|1|32|1000|0.1|bert-base-uncased|500|1e-5|128|42|100000|10000|

### BERT Fine-Tune Logits Generation Scripts

```sh
# Generate BoolQ logits.
python3.8 run_fine_tune_gen_logits.py \
--experiment 1                        \
--model bert                          \
--task boolq                          \
--dataset train                       \
--ckpt 0                              \
--batch_size 128
```

### BERT Fine-Tune Distillation Scripts

```sh
# Fine-tune distillation on BoolQ.
python3.8 run_fine_tune_distill.py \
--experiment 1                     \
--model bert                       \
--task BoolQ                        \
--dataset 1_bert_boolq             \
--num_class 2                      \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
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
# Fine-tune distillation evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_distill_eval.py \
--experiment distill_1                  \
--model bert                            \
--task boolq                            \
--dataset train                         \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_distill_eval.py \
--experiment distill_1                  \
--model bert                            \
--task boolq                            \
--dataset val                           \
--batch_size 128
```

## ALBERT

### ALBERT Fine-Tune Script

```sh
# Fine-tune on BoolQ.
python3.8 run_fine_tune.py  \
--experiment 1              \
--model albert              \
--ptrain_ver albert-base-v2 \
--task boolq                \
--dataset train             \
--num_class 2               \
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
# Fine-tune evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model albert                  \
--task boolq                    \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--model albert                  \
--task boolq                    \
--dataset val                   \
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
|1|0|0|0|0|0|0|8|128|500|0.1|albert-base-v2|250|3e-5|512|42|10000|1000|

### ALBERT Fine-Tune Logits Generation Scripts

```sh
# Generate BoolQ logits.
python3.8 run_fine_tune_gen_logits.py \
--experiment 1                        \
--model albert                        \
--task boolq                          \
--dataset train                       \
--ckpt 0                              \
--batch_size 128
```

### ALBERT Fine-Tune Distillation Scripts

```sh
# Fine-tune distillation on BoolQ.
python3.8 run_fine_tune_distill.py \
--experiment distill_1             \
--model albert                     \
--task boolq                       \
--dataset 1_albert_boolq           \
--num_class 2                      \
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
# Fine-tune distillation evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model albert                  \
--task boolq                    \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_eval.py \
--experiment distill_1          \
--model albert                  \
--task boolq                    \
--dataset dev_mismatched        \
--batch_size 128
```
