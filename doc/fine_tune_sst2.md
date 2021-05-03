# Fine-tune SST2

## Get Data

```sh
# Download SST-2 dataset.
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

# Move SST-2 into data folder.
mv SST-2.zip ./data/fine_tune/SST-2.zip

# Extract SST-2 from zip.
unzip ./data/fine_tune/SST-2.zip -d ./data/fine_tune/

# Remove redundant files.
rm ./data/fine_tune/SST-2.zip
```

## BERT

### BERT Fine-Tune Script

- Train dataset: 67349
  - 1 epoch = 2105 iter (batch size = 32)
- Dev dataset: 872

```sh
# Fine-tune on SST-2.
python3.8 run_fine_tune.py     \
--experiment bert_base_teacher                 \
--model bert                   \
--ptrain_ver bert-base-uncased \
--task sst2                    \
--dataset train                \
--num_class 2                  \
--accum_step 1                 \
--batch_size 32                \
--beta1 0.9                    \
--beta2 0.999                  \｀
--ckpt_step 2105               \
--dropout 0.1                  \
--eps 1e-8                     \
--log_step 500                 \
--lr 3e-5                      \
--max_norm 1.0                 \
--max_seq_len 128              \
--device_id 1                     \
--seed 42                      \
--total_step 6315             \
--warmup_step  2105           \
--weight_decay 0.01
```
