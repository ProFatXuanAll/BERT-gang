# Fine-tune BoolQ 
- [official github link](https://github.com/google-research-datasets/boolean-questions)

## Get Data
```sh
# download
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip

# move to data folder
mv BoolQ.zip ./data/fine_tune/BoolQ.zip

# extract: 'it will create a new folder: ./data/fine_tune/BoolQ/'
unzip ./data/fine_tune/BoolQ.zip -d ./data/fine_tune/ 

# remove redundant files
rm ./data/fine_tune/BoolQ.zip
```


## BERT

### BERT Fine-Tune Script

```sh
python3.8 run_fine_tune.py             \
--experiment 1                         \
--teacher bert                         \
--pretrained_version bert-base-uncased \
--task boolq                            \
--dataset train                        \
--num_class 2                          \
--accumulation_step 8                  \
--batch_size 32                        \
--beta1 0.9                            \
--beta2 0.999                          \
--checkpoint_step 1000                 \
--dropout 0.1                          \
--epoch 10                              \
--eps 1e-8                             \
--learning_rate 1e-5                   \
--max_norm 1.0                         \
--max_seq_len 128                      \
--num_gpu 1                            \
--seed 42                              \
--warmup_step  10000                   \
--weight_decay 0.01
```

### BERT Fine-tune Evaluation Scripts
```sh
# training set
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher bert                  \
--task boolq                     \
--dataset train                 \
--batch_size 64

# validation set
python3.8 run_fine_tune_eval.py \
--experiment 1                  \
--teacher bert                  \
--task boolq                     \
--dataset val           \
--batch_size 64
```