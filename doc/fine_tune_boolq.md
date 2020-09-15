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
--experiment BLarge_3_uncased                 \
--model bert                   \
--ptrain_ver bert-large-uncased \
--task boolq                   \
--dataset train                \
--num_class 2                  \
--accum_step 16                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \
--ckpt_step 250               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 250                 \
--lr 1e-5                      \
--max_norm 1.0                 \
--max_seq_len 256              \
--num_gpu 1                    \
--seed 42                      \
--total_step 2946            \
--warmup_step  295           \
--weight_decay 0.01 \
--amp
```

### BERT Fine-tune Evaluation Scripts

```sh
# Fine-tune evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment BLarge_2                  \
--model bert                    \
--task boolq                    \
--dataset train                 \
--batch_size 128
```

```sh
# Fine-tune evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_eval.py \
--experiment BLarge_2                 \
--model bert                    \
--task boolq                    \
--dataset val                   \
--batch_size 128
```

### BERT Fine-tune Experiment Results

- `*`: indicate training with amp
- Shared configuration

|beta1|beta2|eps|max_norm|weight_decay|
|-|-|-|-|-|
|0.9|0.999|1e-8|1.0|0.01|

- Individual configuration

|ex|train acc|train acc ckpt|val acc|val acc ckpt|accum step|batch|ckpt step|dropout|encoder|log step|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|2|0.995863|5892.ckpt|0.731498|3900.ckpt|1|16|100|0.1|bert-base-uncased|50|1e-5|200|42|5892|590|
|test-amp|0.998727|5000.ckpt|0.723242|4500.ckpt|2|32|500|0.1|bert-base-uncased|500.ckpt|1e-5|256|42|5892|590|
|BLarge_1_amp*|0.992681|2946.ckpt|0.733333|2250.ckpt|16|32|250|0.1|bert-large-cased|250|1e-5|256|42|2946|295|
|BLarge_3_uncased*|0.996075|2946.ckpt|0.788073|2500.ckpt|16|32|250|0.1|bert-large-uncased|250|1e-5|256|42|2946|295|

### BERT Fine-Tune Logits Generation Scripts

```sh
# Generate BoolQ logits.
python3.8 run_fine_tune_gen_logits.py \
--experiment BLarge_uncased_TA_1                        \
--model bert                          \
--task boolq                          \
--dataset train                       \
--ckpt 2946                              \
--batch_size 128
```

### BERT Fine-Tune Distillation Scripts

```sh
# Fine-tune distillation on BoolQ.
python3.8 run_fine_tune_distill.py \
--experiment BLarge_uncased_TA_3                     \
--model bert                       \
--task boolq                        \
--dataset BLarge_uncased_TA_1_bert_boolq            \
--num_class 2                      \
--accum_step 8                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 200                   \
--d_ff 4096                        \
--d_model 1024                      \
--dropout 0.1                      \
--eps 1e-8                         \
--lr 1e-5                          \
--max_norm 1.0                     \
--max_seq_len 256                  \
--num_attention_heads 16           \
--num_gpu 1                        \
--num_hidden_layers 6              \
--seed 42                          \
--total_step 2946                \
--type_vocab_size 2                \
--warmup_step  295               \
--weight_decay 0.01              \
--amp
```

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on BoolQ dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment BLarge_uncased_TA_3                  \
--model bert                            \
--task boolq                            \
--dataset train                         \
--batch_size 128
```

```sh
# Fine-tune distillation evaluation on BoolQ dataset `val`.
python3.8 run_fine_tune_eval.py \
--experiment BLarge_uncased_TA_3                  \
--model bert                            \
--task boolq                            \
--dataset val                           \
--batch_size 128
```

### BERT Fine-tune Distillation Experiment Results

- `*`: indicate distillation with amp
- Shared configuration

|beta1|beta2|eps|max_norm|weight_decay|
|-|-|-|-|-|
|0.9|0.999|1e-8|1.0|0.01|

- Individual configuration

|ex|train acc|train acc ckpt|val acc|val acc ckpt|accum step|batch|ckpt step|dropout|dim_model|dim_FF|num_attn_heads|num_hidden_layers|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|student-2|0.875252|5800.ckpt|0.658716|4600.ckpt|1|16|200|0.1|768|3072|16|6|1e-5|200|42|5892|590|
|BLarge_uncased_distill_1*|0.866554|2946.ckpt|0.673394|2200.ckpt|8|32|200|0.1|1024|4096|16|6|1e-5|256|42|2946|295|
|BLarge_uncased_distill_2*|0.853294|2946.ckpt|0.658716|2000.ckpt|8|32|200|0.1|1024|4096|16|12|1e-5|256|42|2946|295|

### BERT Fine-tune Distillation(initialized from pre-trained teacher) Experiment Results

|ex|train acc|train acc ckpt|val acc|val acc ckpt|accum step|batch|ckpt step|dropout|dim_model|dim_FF|num_attn_heads|num_hidden_layers|lr|max_seq_len|seed|total step|warmup step|start|end|step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BLarge_uncased_distill_3*(attn_W only)|0.922351|2800.ckpt|0.675535|2000.ckpt|8|32|200|0.1|1024|4096|16|6|1e-5|256|42|2946|295|1|6|1|
|BLarge_uncased_distill_5*|0.966797|2946.ckpt|0.659939|2400.ckpt|8|32|200|0.1|1024|4096|16|6|1e-5|256|42|2946|295|1|6|1|
|BLarge_uncased_distill_6|0.934019|2800.ckpt|0.680428|1600.ckpt|8|32|200|0.1|1024|4096|16|12|1e-5|256|42|2946|295|1|24|2|
|BLarge_uncased_distill_7*|0.955023|2946.ckpt|0.685321|2200.ckpt|8|32|200|0.1|1024|4096|16|6|1e-5|256|42|2946|295|1|24|4|
|BLarge_uncased_distill_8*|0.908773|1470.ckpt|0.679205|1400.ckpt|16|64|200|0.1|1024|4096|16|6|1e-5|256|42|1470|147|1|24|4|

### BERT Fine-tune Distillation with TA Experiments Results

|ex|train acc|train acc ckpt|val acc|val acc ckpt|accum step|batch|ckpt step|dropout|dim_model|dim_FF|num_attn_heads|num_hidden_layers|lr|max_seq_len|seed|total step|warmup step|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BLarge_uncased_TA_1*|0.853294|2946.ckpt|0.658716|2000.ckpt|8|32|200|0.1|1024|4096|16|12|1e-5|256|42|2946|295|
|BLarge_uncased_TA_2*|0.825607|2946.ckpt|0.659633|2000.ckpt|16|32|200|0.1|1024|4096|16|21|1e-5|256|42|2946|295|
|BLarge_uncased_TA_3*|0.853294|2946.ckpt|0.656575|2800.ckpt|8|32|200|0.1|1024|4096|16|12|1e-5|256|42|2946|295|



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
