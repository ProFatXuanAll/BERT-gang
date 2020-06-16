from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AlbertConfig, AlbertTokenizer

import dataset
import albert_fine_tune

EXPERIMENT_NO = 0
BATCH_SIZE = 32
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

writer = torch.utils.tensorboard.SummaryWriter(PATH['log'])

checkpoint_pattern = r'(\d+)\.pt'
all_checkpoint = sorted(
    os.listdir(PATH['checkpoint']),
    key=lambda checkpoint: int(
        re.match(checkpoint_pattern, checkpoint).group(1))
)

print(f'======MNLI ALBERT FINE-TUNE EXPERIMENT {EXPERIMENT_NO} EVALUATION======')
train_max_acc = 0
train_max_acc_checkpoint = None
dev_matched_max_acc = 0
dev_matched_max_acc_checkpoint = None
dev_mismatched_max_acc = 0
dev_mismatched_max_acc_checkpoint = None
for checkpoint in all_checkpoint:
    model = albert_fine_tune.AlbertFineTuneModel(
        in_features=config.hidden_size,
        out_features=dataset.MNLI.num_label(),
        pretrained_version='albert-base-v2',
        dropout_prob=config.hidden_dropout_prob
    )
    model.load_state_dict(torch.load(f'{PATH["checkpoint"]}/{checkpoint}'))
    model = model.to(device)
    model.eval()

    for file_name in ['train', 'dev_matched', 'dev_mismatched']:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset.MNLI(file_name),
            batch_size=BATCH_SIZE,
            collate_fn=dataset.MNLI.create_collate_fn(tokenizer),
            shuffle=True
        )

        print(f'======{checkpoint} EVALUATE {file_name}======')
        all_label = []
        all_pred_label = []
        with torch.no_grad():
            for (input_ids,
                 attention_mask,
                 token_type_ids,
                 label) in tqdm(eval_dataloader):

                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)

                pred_label = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                ).to('cpu').argmax(dim=-1)

                all_label.extend(label.tolist())
                all_pred_label.extend(pred_label.tolist())

        writer.add_scalar(
            f'{file_name} accuracy',
            accuracy_score(all_label, all_pred_label),
            int(re.match(checkpoint_pattern, checkpoint).group(1))
        )

        if file_name == 'train':
            train_acc = accuracy_score(all_label, all_pred_label)
            if train_max_acc <= train_acc:
                train_max_acc = train_acc
                train_max_acc_checkpoint = checkpoint

        if file_name == 'dev_matched':
            dev_matched_acc = accuracy_score(all_label, all_pred_label)
            if dev_matched_max_acc <= dev_matched_acc:
                dev_matched_max_acc = dev_matched_acc
                dev_matched_max_acc_checkpoint = checkpoint

        if file_name == 'dev_mismatched':
            dev_mismatched_acc = accuracy_score(all_label, all_pred_label)
            if dev_mismatched_max_acc <= dev_mismatched_acc:
                dev_mismatched_max_acc = dev_mismatched_acc
                dev_mismatched_max_acc_checkpoint = checkpoint

writer.close()
print(
    f'======FINISH MNLI ALBERT FINE-TUNE EXPERIMENT {EXPERIMENT_NO} EVALUATION======')
print(f'train best checkpoint: {train_max_acc_checkpoint}')
print(f'train best accuracy: {train_max_acc}')
print(f'dev matched best checkpoint: {dev_matched_max_acc_checkpoint}')
print(f'dev matched best accuracy: {dev_matched_max_acc}')
print(f'dev mismatched best checkpoint: {dev_mismatched_max_acc_checkpoint}')
print(f'dev mismatched best accuracy: {dev_mismatched_max_acc}')
