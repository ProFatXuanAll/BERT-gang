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
- 1 epochs = 3274 iters
  - under batch size 32

### BERT Fine-Tune Script

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

### BERT-PKD Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo pkd-user-defined                          \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment pkd-user-defined-1            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--type_vocab_size 2                \
--seed 42                          \
--warmup_step  1309               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 10                  \
--mu 100                           \
--soft_weight 0.5
```

### AKD-BERT Fine-Tune Distillation Scripts with Multi-GPU

```sh
python3.8 run_fine_tune_distill_mgpu.py \
--kd_algo akd                          \
--teacher_exp teacher_base                \
--tmodel bert                      \
--tckpt  9822 \
--experiment gate_network_new_obj1            \
--model bert                       \
--task qnli                        \
--accum_step 1                     \
--batch_size 32                    \
--beta1 0.9                        \
--beta2 0.999                      \
--ckpt_step 1000                   \
--d_ff 3072                        \
--d_model 768                      \
--dropout 0.1                      \
--eps 1e-8                         \
--log_step 100                     \
--lr 5e-5                          \
--max_norm 1.0                     \
--num_attention_heads 12           \
--num_hidden_layers 6              \
--total_step 13096                \
--type_vocab_size 2                \
--warmup_step  1309               \
--weight_decay 0.01                \
--device_id 0                      \
--tdevice_id 0                     \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                \
--mu 100                           \
--use_hidden_loss                  \
--use_classify_loss
```

### BERT Fine-Tune Evaluation Scripts

```sh
# Fine-tune evaluation on QNLI dataset `train`.
python3.8 run_fine_tune_eval.py \
--experiment pkd_even_65                 \
--model bert                    \
--task qnli                     \
--dataset train                 \
--batch_size 512                \
--device_id 1
```

```sh
# Fine-tune evaluation on QNLI dataset `dev`.
python3.8 run_fine_tune_eval.py \
--experiment pkd_even_65                 \
--model bert                    \
--task qnli                     \
--dataset dev           \
--batch_size 512 \
--device_id 1
```

### Plot CLS embedding of last Transformer block

```sh
python3.8 plot_CLS_embedding.py  \
--ckpt 6548                     \
--experiment SCL_6_ce             \
--model bert                     \
--task qnli                      \
--dataset dev            \
--batch_size 256                 \
--device_id 1
```
