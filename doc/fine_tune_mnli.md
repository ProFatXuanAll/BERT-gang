# Fine-tune MNLI

## Get Data

```sh
# download
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip

# move to data folder
mv multinli_1.0.zip ./data/fine_tune/mnli.zip

# extract
unzip ./data/fine_tune/mnli.zip -d ./data/fine_tune/mnli

# format file names
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl ./data/fine_tune/mnli/dev_matched.jsonl
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl ./data/fine_tune/mnli/dev_mismatched.jsonl
mv ./data/fine_tune/mnli/multinli_1.0/multinli_1.0_train.jsonl ./data/fine_tune/mnli/train.jsonl

# remove redundant files
rm -rf ./data/fine_tune_data/mnli/__MACOSX
rm -rf ./data/fine_tune_data/mnli/multinli_1.0
rm ./data/fine_tune/mnli.zip
```

## BERT

### BERT Fine-Tune Script

```sh
python3.8 run_fine_tune.py             \
--experiment 1                         \
--teacher bert                         \
--pretrained_version bert-base-uncased \
--task mnli                            \
--dataset train                        \
--num_class 3                          \
--accumulation_step 1                  \
--batch_size 32                        \
--beta1 0.9                            \
--beta2 0.999                          \
--checkpoint_step 1000                 \
--dropout 0.1                          \
--epoch 3                              \
--eps 1e-8                             \
--learning_rate 3e-5                   \
--max_norm 1.0                         \
--max_seq_len 128                      \
--num_gpu 1                            \
--seed 42                              \
--warmup_step  10000                   \
--weight_decay 0.01
```

### BERT Fine-Tune Evaluation Scripts

```sh
# train
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher bert                  \
--task mnli                     \
--dataset train                 \
--batch_size 64

# dev_matched
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher bert                  \
--task mnli                     \
--dataset dev_matched           \
--batch_size 64

# dev_mismatched
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher bert                  \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 64
```

### BERT Fine-Tune Experiment Results

|ex|train acc|dev-m acc|dev-m acc ckpt|dev-mm acc|dev-mm acc ckpt|encoder|epoch|lr|batch|accum step|beta1|beta2|eps|weight decay|warmup step|dropout|max_norm|max_seq_len|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0.903957|0.826693|35000|0.833299|33000|bert-base-cased|3|3e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|3|0.905185|0.827305|36000|0.835435|35000|bert-base-cased|3|3e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|5|0.905325|0.825776|33000|0.837571|35000|bert-base-cased|3|2e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|6|0.904507|0.828527|33000|0.836045|35000|bert-base-uncased|3|3e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|7|0.902900|0.830259|36000|0.835333|36000|bert-base-uncased|3|2e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|

### BERT Fine-Tune Soft-Target Generation Scripts

```sh
python3.8 run_fine_tune_gen_soft_target.py \
--experiment 1                             \
--teacher bert                             \
--task mnli                                \
--dataset train                            \
--ckpt 0                                   \
--batch_size 64
```

### BERT Fine-Tune Distillation Scripts

```sh
python3.8 run_fine_tune_distill.py     \
--experiment 1                         \
--student bert                         \
--task mnli                            \
--dataset bert-mnli-1                  \
--num_class 3                          \
--accumulation_step 1                  \
--batch_size 32                        \
--beta1 0.9                            \
--beta2 0.999                          \
--checkpoint_step 1000                 \
--d_emb 768                            \
--d_ff 3072                            \
--d_model 768                          \
--dropout 0.1                          \
--epoch 3                              \
--eps 1e-8                             \
--learning_rate 3e-5                   \
--max_norm 1.0                         \
--max_seq_len 128                      \
--num_attention_heads 16               \
--num_gpu 1                            \
--num_hidden_layers 6                  \
--seed 42                              \
--type_vocab_size 2                    \
--warmup_step  10000                   \
--weight_decay 0.01
```

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# train
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student bert                          \
--task mnli                             \
--dataset train                         \
--batch_size 64

# dev_matched
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student bert                          \
--task mnli                             \
--dataset dev_matched                   \
--batch_size 64

# dev_mismatched
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student bert                          \
--task mnli                             \
--dataset dev_mismatched                \
--batch_size 64
```

## ALBERT

### ALBERT Fine-Tune Script

```sh
python3.8 run_fine_tune.py          \
--experiment 1                      \
--teacher albert                    \
--pretrained_version albert-base-v2 \
--task mnli                         \
--dataset train                     \
--num_class 3                       \
--accumulation_step 8               \
--batch_size 128                    \
--beta1 0.9                         \
--beta2 0.999                       \
--checkpoint_step 1000              \
--dropout 0.1                       \
--epoch 3                           \
--eps 1e-8                          \
--learning_rate 3e-5                \
--max_norm 1.0                      \
--max_seq_len 512                   \
--num_gpu 1                         \
--seed 42                           \
--warmup_step  1000                 \
--weight_decay 0.01
```

### ALBERT Fine-Tune Evaluation Scripts

```sh
# train
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher albert                \
--task mnli                     \
--dataset train                 \
--batch_size 128

# dev_matched
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher albert                \
--task mnli                     \
--dataset dev_matched           \
--batch_size 128

# dev_mismatched
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher albert                \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 128
```

### ALBERT Experiment Results

|ex|train acc|dev-m acc|dev-m acc ckpt|dev-mm acc|dev-mm acc ckpt|encoder|epoch|lr|batch|accum step|beta1|beta2|eps|weight decay|warmup step|dropout|max_norm|max_seq_len|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|0|0.890685|0.841874|8900|0.849877|9100|albert-base-v2|3|3e-5|128|8|0.9|0.999|1e-8|0.01|1000|0.1|1.0|512|42|
|1|0.890402|0.839836|8200|0.845809|9100|albert-base-v2|3|3e-5|128|8|0.9|0.999|1e-8|0.01|1000|0|1.0|512|42|

### ALBERT Fine-Tune Soft-Target Generation Scripts

```sh
python3.8 run_fine_tune_gen_soft_target.py \
--experiment 1                             \
--teacher albert                           \
--task mnli                                \
--dataset train                            \
--ckpt 0                                   \
--batch_size 128
```

### ALBERT Fine-Tune Distillation Scripts

```sh
python3.8 run_fine_tune_distill.py     \
--experiment 1                         \
--student albert                       \
--task mnli                            \
--dataset albert-mnli-1                \
--num_class 3                          \
--accumulation_step 1                  \
--batch_size 32                        \
--beta1 0.9                            \
--beta2 0.999                          \
--checkpoint_step 1000                 \
--d_emb 128                            \
--d_ff 3072                            \
--d_model 768                          \
--dropout 0.1                          \
--epoch 3                              \
--eps 1e-8                             \
--learning_rate 3e-5                   \
--max_norm 1.0                         \
--max_seq_len 128                      \
--num_attention_heads 16               \
--num_gpu 1                            \
--num_hidden_layers 6                  \
--seed 42                              \
--type_vocab_size 2                    \
--warmup_step  10000                   \
--weight_decay 0.01
```

### ALBERT Fine-Tune Distillation Evaluation Scripts

```sh
# train
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student albert                        \
--task mnli                             \
--dataset train                         \
--batch_size 64

# dev_matched
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student albert                        \
--task mnli                             \
--dataset dev_matched                   \
--batch_size 64

# dev_mismatched
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student albert                        \
--task mnli                             \
--dataset dev_mismatched                \
--batch_size 64
```
