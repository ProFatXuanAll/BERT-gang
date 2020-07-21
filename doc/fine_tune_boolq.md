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
--experiment 1                  
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
|6|0.675719|0.651682|2945.ckpt|bert-large-uncased|10|1e-5|32|32|0.9|0.999|1e-8|0.01|10000|0.1|1.0|128|42|
|**7**|0.836109|**0.764832**|2946.ckpt|bert-large-uncased|10|1e-5|32|16|0.9|0.999|1e-08|0.01|10000|0.1|1.0|256|42|
|8|0.623104|0.621713|2945.ckpt|ber-large-cased|10|1e-5|32|32|0.9|0.999|1e-08|0.01|10000|0.1|1.0|256|42|
|9|0.623104|0.690826|1473.ckpt|bert-large-cased|10|1e-5|64|32|0.9|0.999|1e-08|0.01|10000|0.1|1.0|256|42|
|10|0.801209|0.720183|2946.ckpt|ber-large-cased|10|1e-5|32|16|0.9|0.999|1e-08|0.01|10000|0.1|1.0|256|42|

### BERT Fine-Tune Soft-Target Generation Scripts
```sh
python3.8 run_fine_tune_gen_soft_target.py \
--experiment 1                             \
--teacher bert                             \
--task boolq                                \
--dataset train                            \
--ckpt 0                                   \
--batch_size 64
```

### BERT Fine-Tune Distillation Scripts
```sh
python3.8 run_fine_tune_distill.py     \
--experiment 1                         \
--student bert                         \
--task boolq                           \
--dataset ex-7-teacher-bert-task-boolq  \
--num_class 2                          \
--accumulation_step 1                  \
--batch_size 32                        \
--beta1 0.9                            \
--beta2 0.999                          \
--checkpoint_step 200                 \
--d_emb 768                            \
--d_ff 3072                            \
--d_model 768                          \
--dropout 0.1                          \
--epoch 3                              \
--eps 1e-8                             \
--learning_rate 1e-5                   \
--max_norm 1.0                         \
--max_seq_len 256                      \
--num_attention_heads 16               \
--num_gpu 1                            \
--num_hidden_layers 6                  \
--seed 42                              \
--type_vocab_size 2                    \
--warmup_step  10000                   \
--weight_decay 0.01
```

### BERT Fine-Tune Distillation Evaluation Scripts
```sh
# train
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student bert                          \
--task boolq                             \
--dataset train                         \
--batch_size 64

# validation
python3.8 run_fine_tune_distill_eval.py \
--experiment 1                          \
--student bert                          \
--task boolq                             \
--dataset val                         \
--batch_size 64
```
### BERT Fine-tune Distillation Experiment Results
|ex|train acc|val acc|val acc ckpt|student|num_hidden_layer|dim_model|dim_FFN|num_attn_head|epoch|lr|batch|accum step|beta1|beta2|eps|weight decay|warmup step|dropout|max_norm|max_seq_len|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|0.623104|0.621713|2950.ckpt|BERT|6|768|3072|16|10|1e-5|32|1|0.9|0.999|1e-8|0.01|10000|0.1|1.0|256|42|

