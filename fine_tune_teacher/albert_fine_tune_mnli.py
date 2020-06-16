from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm
from transformers import AlbertConfig, AlbertTokenizer, get_linear_schedule_with_warmup

import dataset
import albert_fine_tune

EXPERIMENT_NO = 0
BATCH_SIZE = 128
ACCUMULATION_STEP = 128
EPOCH = 3
DROPOUT = 0.1
LEARNING_RATE = 3e-5
MAX_GRAD_NORM = 1.0
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_STEP = 1000
LOG_STEP = 100
SEED = 777

PATH = {}
PATH['data'] = os.path.abspath(
    f'{os.path.abspath(__file__)}/../../data'
)
PATH['fine_tune_data'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_data'
)
PATH['experiment'] = os.path.abspath(
    f'{PATH["data"]}/fine_tune_experiment/mnli/albert_experiment_{EXPERIMENT_NO}'
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

config = AlbertConfig.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = albert_fine_tune.AlbertFineTuneModel(
    in_features=config.hidden_size,
    out_features=dataset.MNLI.num_label(),
    pretrained_version='albert-base-v2',
    dropout_prob=DROPOUT
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

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': WEIGHT_DECAY,
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
    },
]
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2),
    eps=EPS
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEP,
    num_training_steps=int(np.ceil(len(train_dataset) / BATCH_SIZE)) * EPOCH
)
objective = nn.CrossEntropyLoss()

print(f'======MNLI ALBERT FINE-TUNE EXPERIMENT {EXPERIMENT_NO}======')
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()

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
