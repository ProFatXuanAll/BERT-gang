from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils
import torch.utils.data
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

import dataset
import bert_fine_tune

EXPERIMENT_NO = 0
BATCH_SIZE = 32
SEED = 777
BEST_CHECKPOINT = 36000

PATH = {}
PATH['data'] = os.path.abspath(
    f'{os.path.abspath(__file__)}/../../data'
)
PATH['fine_tune_data'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_data'
)
PATH['experiment'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_experiment/mnli/bert_experiment_{EXPERIMENT_NO}'
)
PATH['best_checkpoint'] = os.path.abspath(
    f'{PATH["experiment"]}/checkpoint/{BEST_CHECKPOINT}.pt'
)

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config = BertConfig.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = bert_fine_tune.BertFineTuneModel(
    in_features=config.hidden_size,
    out_features=dataset.MNLI.num_label(),
    pretrained_version='bert-base-cased',
    dropout_prob=config.hidden_dropout_prob
)
model.load_state_dict(torch.load(PATH['best_checkpoint']))
model = model.to(device)
model.eval()

print(
    f'======MNLI BERT FINE-TUNE EXPERIMENT {EXPERIMENT_NO} CONSTRUCT DISTILLATION DATA======')

train_dataset = dataset.MNLI('train')
collate_fn = dataset.MNLI.create_collate_fn(tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=False
)

with torch.no_grad():
    index_shift_counter = 0
    for (input_ids,
         attention_mask,
         token_type_ids,
         label) in tqdm(train_dataloader):

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        soft_target = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).detach().to('cpu')

        for index in range(soft_target.size()[0]):
            train_dataset.add_soft_target(
                index + index_shift_counter,
                soft_target[index].tolist()
            )

        index_shift_counter += soft_target.size()[0]

train_dataset.save_distillation_data(
    'bert',
    EXPERIMENT_NO
)
