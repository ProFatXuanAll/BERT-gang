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

### BERT Fine-tune Experiment Results
|ex|train acc|val acc|val acc ckpt|encoder|epoch|lr|batch|accum step|beta1|beta2|eps|weight decay|warmup step|dropout|max_norm|max_seq_len|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0.868457|0.717125|1000.ckpt|bert-base-uncased|10|1e-5|32|8|0.9|0.999|1e-08|0.01|10000|0.1|1.0|128|42|
|2|0.871857|0.696942|2946.ckpt|bert-base-cased|10|1e-5|32|8|0.9|0.999|1e-8|0.01|10000|0.1|1.0|128|42|
|3|0.658004|0.587768|2945.ckpt|bert-base-uncased|10|1e-5|32|8|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|4|0.814151|0.704587|1473.ckpt|bert-base-uncased|10|1e-5|64|32|0.9|0.999|1e-8|0.01|10000|0.1|1.0|512|42|
|5|0.864750|0.697859|1473.ckpt|bert-base-uncased|10|1e-5|64|16|0.9|0.999|1e-8|0.01|10000|0.1|1.0|128|42|