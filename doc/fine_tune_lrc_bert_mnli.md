# LRC-BERT training

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

- 3 classification
- train: 392702
- dev_mismatched: 9832
- dev_matched: 9815

## BERT

- Batch Size: 16, 32
- Learning Rate: 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4
- Dropout: 0.1
- 1 epochs
  - 12272 iters (under batch size 32)
  - 24543 iters (under batch size 16)

## Train LRC BERT Script

```sh
python3.8 train_lrc_bert.py \
--teacher_exp test                \
--tmodel bert                      \
--tckpt  36816 \
--experiment LRC_debug            \
--model bert                       \
--task mnli                        \
--accum_step 1                     \
--batch_size 16                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 10000                   \
--d_ff 1200                        \
--d_model 312                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 4              \
--total_step 441774                \
--type_vocab_size 2                \
--warmup_step  44177               \
--weight_decay 0.01                \
--device_id 1                      \
--neg_num 15                    \
--contrast_steps 353419           \
--softmax_temp 1.1                \
--soft_label_weight 1        \
--hard_label_weight 3        \
--nce_cos_weight 1
```
