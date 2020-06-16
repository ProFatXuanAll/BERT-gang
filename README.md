# BERT-gang

Using Knowledge Distillation to learn from BERT-like models.

## Setup

```sh
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
