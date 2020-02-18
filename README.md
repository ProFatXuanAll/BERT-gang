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
  'input_ids': [101, 146, 1176, 4629, 194, 1358, 119, 102, 146, 1176, 6363, 119, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
  'token_bound_ids': [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}
```
