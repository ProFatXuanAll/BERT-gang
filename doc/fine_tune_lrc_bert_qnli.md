# LRC-BERT training

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
- 1 epochs
  - 3274 iters (under batch size 32)
  - 6545 iters (under batch size 16)

## Train LRC BERT Script

```sh
python3.8 train_lrc_bert.py \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment LRC_debug            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 16                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 5000                   \
--d_ff 1200                        \
--d_model 312                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 4              \
--total_step 117810                \
--type_vocab_size 2                \
--warmup_step  11781               \
--weight_decay 0.01                \
--device_id 1                      \
--neg_num 15                    \
--contrast_steps 94248           \
--softmax_temp 1.1                \
--soft_label_weight 1        \
--hard_label_weight 3        \
--nce_cos_weight 1
```
