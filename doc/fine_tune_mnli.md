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
--experiment bert_base_teacher                 \
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

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment bert_base_teacher                  \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 256                \
--device_id 0
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment bert_base_teacher                  \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment bert_base_teacher                  \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 512                \
--device_id 1
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

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--teacher_exp test                \
--tmodel bert                      \
--tckpt  36816 \
--experiment PKD            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 12272                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 49088                \
--type_vocab_size 2                \
--warmup_step  4908               \
--weight_decay 0.01                \
--device_id 1                      \
--tdevice_id 1                     \
--mu 100                           \
--use_hidden_loss                  \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--use_classify_loss                \
--cls_steps 0                   \
--ce_weight 1                      \
--scl_temp 1

```

#### Fine-Tune Distillation with Contrastive learning

```sh
python3.8 run_fine_tune_contrast_distill.py \
--teacher_exp test                \
--tmodel bert                      \
--tckpt  36816 \
--experiment contrast_by_label_1             \
--model bert                       \
--task mnli                        \
--accum_step 4                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 122720                \
--type_vocab_size 2                \
--warmup_step  8181               \
--weight_decay 0.01                \
--device_id 1                      \
--membank_device 1                 \
--neg_num 10                    \
--two_stage                        \
--teacher_device 1                 \
--contrast_steps 98176
```

#### Fine-Tune Distillation with Contrastive learning layer wise

```sh
python3.8 run_layerwise_contrast_distill.py \
--teacher_exp test                \
--tmodel bert                      \
--tckpt  36816 \
--experiment debug             \
--model bert                       \
--task mnli                        \
--accum_step 2                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 5000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 122720                \
--type_vocab_size 2                \
--warmup_step  30000               \
--weight_decay 0.01                \
--device_id 1                      \
--neg_num 20                    \
--contrast_steps 0           \
--contrast_temp 0.07             \
--softmax_temp 1                \
--soft_label_weight 0.2         \
--defined_by_label
```

### Train SCL independently

```sh
python3.8 train_scl_from_ckpt.py \
--experiment SCL_1               \
--src_experiment MSE_base        \
--src_ckpt 24544                  \
--model bert                     \
--task mnli                      \
--device_id 1                    \
--scl_temp 0.1                   \
--accum_step 1                   \
--batch_size 32                  \
--ckpt_step 2000                 \
--log_step 100                   \
--lr 5e-5                        \
--total_step 24544                \
--warmup_step 2454
```

### Train KD (output logits CE) independently

```sh
python3.8 train_kd_from_ckpt.py \
--teacher_exp test      \
--tmodel bert                   \
--tckpt 36816                    \
--experiment SCL_1_ce           \
--src_experiment SCL_1          \
--src_ckpt 24544                 \
--model bert                    \
--task mnli                     \
--device_id 1                   \
--tdevice_id 1                  \
--softmax_temp 10               \
--soft_weight 0.5               \
--accum_step 1                   \
--batch_size 32                  \
--ckpt_step 1000                 \
--log_step 100                   \
--lr 1e-5                        \
--total_step 24544                \
--warmup_step 2454
```

### BERT Fine-Tune Distillation Evaluation Scripts

```sh
# Fine-tune distillation evaluation on MNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment PKD          \
--model bert                    \
--task mnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 0                   \
--ckpt 98176
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_matched`.
python3.8 run_fine_tune_eval.py \
--experiment PKD          \
--model bert                    \
--task mnli                     \
--dataset dev_matched           \
--batch_size 512                \
--device_id 1                   \
--ckpt 98176
```

```sh
# Fine-tune distillation evaluation on MNLI dataset `dev_mismatched`.
python3.8 run_fine_tune_eval.py \
--experiment PKD          \
--model bert                    \
--task mnli                     \
--dataset dev_mismatched        \
--batch_size 512 \
--device_id 1 \
--ckpt 98176
```

### Build memory bank

```sh
python3.8 build_membank.py \
--experiment teacher_base \
--model bert \
--task mnli \
--dataset train \
--ckpt 36816 \
--batch_size 256 \
--device_id 0 \
--layer_wise
```

### Build logits bank

```sh
python3.8 build_logitsbank.py \
--experiment test \
--model bert \
--task mnli \
--dataset train \
--ckpt 36816 \
--batch_size 256 \
--device_id 1
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
