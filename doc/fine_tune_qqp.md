# Fine-tune QQP

## Get Data

```sh
# Download QQP dataset.
wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip

# Move QQP into data folder.
mv QQP-clean.zip ./data/fine_tune/QQP-clean.zip

# Extract QQP from zip.
unzip ./data/fine_tune/QQP-clean.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/QQP-clean.zip
```

## BERT Fine-Tune

### BERT Fine-Tune Script

- Train dataset: 363846
  - 1 epoch = 11370 iter (batch size = 32)
- Dev dataset: 40430

```sh
python run_fine_tune.py   \
--experiment teacher_base   \
--model bert                \
--ptrain_ver bert-base-uncased   \
--task qqp                       \
--dataset train                   \
--num_class 2                      \
--accum_step 1                    \
--batch_size 32                   \
--beta1 0.9                       \
--beta2 0.999                     \
--ckpt_step 11370                 \
--dropout 0.1                     \
--eps 1e-8                        \
--log_step 500                    \
--lr 2e-5                         \
--max_norm 1.0                    \
--max_seq_len 128                 \
--device_id 1                     \
--seed 42                         \
--total_step 34110                \
--warmup_step 10233               \
--weight_decay 0.01
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QQP dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment teacher_base                 \
--model bert                    \
--task qqp                     \
--dataset train                 \
--batch_size 256                \
--device_id 0
```

```sh
# Fine-tune evaluation on QQP dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment teacher_base                 \
--model bert                    \
--task qqp                     \
--dataset dev           \
--batch_size 512 \
--device_id 0
```
