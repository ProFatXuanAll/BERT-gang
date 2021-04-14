# Fine-tune QNLI

## Get Data

```sh
# Download QNLI dataset.
wget https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip

# Move QNLI into data folder.
mv QNLIv2.zip ./data/fine_tune/qnli.zip

# Extract QNLI from zip.
unzip ./data/fine_tune/qnli.zip -d ./data/fine_tune

# Remove redundant files.
rm ./data/fine_tune/qnli.zip
```

- binary classification
- train: 104743
- dev: 5463

## BERT

- Batch Size: 16, 32
- Learning Rate: 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4
- Dropout: 0.1

### BERT Fine-Tune Script

- 1 epochs = 3274 iters
  - under batch size 32

```sh
python3.8 run_fine_tune.py     \
--experiment teacher_base       \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task qnli                    \
--dataset train                \
--num_class 2                  \
--accum_step 2                 \
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
--device_id 0                     \
--seed 42                      \
--total_step 9822             \
--warmup_step 3274            \
--weight_decay 0.01
```

### Fine-Tune Distillation with layer wise Contrastive learning

```sh
python3.8 run_layerwise_contrast_distill.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment Contrast_by_sample            \
--model bert                       \
--task qnli                        \
--accum_step 2                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 2000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 500                     \
--lr 3e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 26192                \
--type_vocab_size 2                \
--warmup_step  2619               \
--weight_decay 0.01                \
--device_id 1                      \
--neg_num 20                    \
--contrast_steps 0           \
--contrast_temp 0.1             \
--softmax_temp 1                \
--soft_label_weight 0.2        \
--contrast_weight 0.7         \
--defined_by_label
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment Contrast_by_sample                 \
--model bert                    \
--task qnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on QNLI dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment Contrast_by_sample                 \
--model bert                    \
--task qnli                     \
--dataset dev           \
--batch_size 512 \
--device_id 1
```

### Build memory bank

```sh
python3.8 build_membank.py \
--experiment teacher_base \
--model bert \
--task qnli \
--dataset train \
--ckpt 9822 \
--batch_size 256 \
--device_id 1
```

### Build logits bank

```sh
python3.8 build_logitsbank.py \
--experiment teacher_base \
--model bert \
--task qnli \
--dataset train \
--ckpt 9822 \
--batch_size 256 \
--device_id 0
```

### Plot CLS embedding of last Transformer block

```sh
python3.8 plot_CLS_embedding.py  \
--ckpt 13096                     \
--experiment MSE_init_from_pre_trained             \
--model bert                     \
--task qnli                      \
--dataset dev            \
--batch_size 256                 \
--device_id 0
```
