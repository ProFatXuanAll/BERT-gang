# BERT-gang

Using Knowledge Distillation to learn from BERT-like models.

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
  'segment_a': 'I like hung yu.',
  'segment_b': 'I like dogs.',
}
```

- `albert.pickle`, `bert.pickle`, `roberta.pickle` files structure

```py
{
  # output tensor from model
  # shape: (seq_len, hid_dim)
  'output_embeds': [[0.0, ...,] [0.0,...], ...],

  # segment indicator
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],

  # token boundary indicator
  'token_bound_ids': [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}
```
