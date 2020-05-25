# BERT-gang

Using Knowledge Distillation to learn from BERT-like models.

## Setup

```
# clone the project
git clone https://github.com/ProFatXuanAll/BERT-gang

# change to the project directory
cd BERT-gang

# create data folder
mkdir data data/fine_tune_data
```

## Pre-train data

- Directory structure

```text
data
╠═ .gitignore
╠═ example.txt
╚═ pre-train
   ╠═ 1
   ║ ╠═ original.pickle
   ║ ╠═ albert.pickle
   ║ ╠═ bert.pickle
   ║ ╚═ roberta.pickle
   ╠═ 2
   ║ ╠═ original.pickle
   ║ ╠═ albert.pickle
   ║ ╠═ bert.pickle
   ║ ╚═ roberta.pickle
   ╚═ ...
```

- `original.pickle` files structure

```py
{
  'segment_a': 'Example segment 1.',
  'segment_b': 'Example segment 2.',
}
```

- `albert.pickle`, `bert.pickle`, `roberta.pickle` files structure

```py
# output tensor from model
# shape: (seq_len, hid_dim)
[[0.0, ...,] [0.0,...], ...]
```

## Fine-tune data

### MNLI

```sh
# download
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip ./data/fine_tune_data/mnli.zip

# extract
unzip ./data/fine_tune_data/mnli.zip -d ./data/fine_tune_data/mnli

# format file names
mv ./data/fine_tune_data/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl ./data/fine_tune_data/mnli/dev_matched.jsonl
mv ./data/fine_tune_data/mnli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl ./data/fine_tune_data/mnli/dev_mismatched.jsonl
mv ./data/fine_tune_data/mnli/multinli_1.0/multinli_1.0_train.jsonl ./data/fine_tune_data/mnli/train.jsonl

# remove redundant files
rm -rf ./data/fine_tune_data/mnli/__MACOSX
rm -rf ./data/fine_tune_data/mnli/multinli_1.0
rm ./data/fine_tune_data/mnli.zip
```
