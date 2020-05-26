from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

import dataset
import bert_fine_tune

EXPERIMENT_NO = 1
BATCH_SIZE = 32
ACCUMULATION_STEP = 8
EPOCH = 3
LEARNING_RATE = 3e-5
LOG_STEP = 1000
SEED = 777

PATH = {}
PATH['data'] = os.path.abspath(
    f'{os.path.abspath(__file__)}/../../data'
)
PATH['fine_tune_data'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_data'
)
PATH['experiment'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_experiment/mnli/experiment_{EXPERIMENT_NO}'
)
PATH['log'] = os.path.abspath(f'{PATH["experiment"]}/log')
PATH['checkpoint'] = os.path.abspath(f'{PATH["experiment"]}/checkpoint')

if not os.path.exists(PATH['experiment']):
    os.makedirs(PATH['experiment'])
if not os.path.exists(PATH['log']):
    os.makedirs(PATH['log'])
if not os.path.exists(PATH['checkpoint']):
    os.makedirs(PATH['checkpoint'])

device = torch.device('cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)

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
    pretrained_version='bert-base-cased'
)
model = model.to(device)

train_dataset = dataset.MNLI('train')
collate_fn = dataset.MNLI.create_collate_fn(tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE // ACCUMULATION_STEP,
    collate_fn=collate_fn,
    shuffle=True
)

writer = torch.utils.tensorboard.SummaryWriter(PATH['log'])

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE)
objective = nn.CrossEntropyLoss()

print(f'======MNLI BERT FINE-TUNE EXPERIMENT {EXPERIMENT_NO}======')
step_counter = 0
for epoch in range(EPOCH):
    print(f'======EPOCH {epoch}======')
    accumulation_loss = 0
    for (input_ids,
         attention_mask,
         token_type_ids,
         label) in tqdm(train_dataloader):

        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        loss = objective(
            model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            ),
            label
        )

        accumulation_loss += loss / ACCUMULATION_STEP
        step_counter += 1

        actual_step = step_counter // ACCUMULATION_STEP

        if step_counter % ACCUMULATION_STEP == 0:
            accumulation_loss.backward()

            optimizer.step()

            writer.add_scalar('loss',
                              accumulation_loss.item(),
                              actual_step)

            accumulation_loss.detach()
            del accumulation_loss
            accumulation_loss = 0

        if actual_step % LOG_STEP == 0:
            torch.save(
                model.state_dict(),
                f'{PATH["checkpoint"]}/{actual_step}.pt'
            )

        loss.detach()
        input_ids.detach()
        attention_mask.detach()
        token_type_ids.detach()
        label.detach()
        del loss
        del input_ids
        del attention_mask
        del token_type_ids
        del label

        torch.cuda.empty_cache()

writer.close()
torch.save(
    model.state_dict(),
    f'{PATH["checkpoint"]}/{actual_step}.pt'
)
