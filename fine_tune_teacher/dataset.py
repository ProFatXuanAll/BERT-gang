"""
All fine-tune dataset should go to this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import torch
import torch.utils
import torch.utils.data

DATA_PATH = os.path.abspath(
    f'{os.path.abspath(__file__)}/../../data'
)
FINE_TUNE_DATA_PATH = os.path.abspath(
    f'{DATA_PATH}/fine_tune_data'
)
FINE_TUNE_DISTILLATION_DATA_PATH = os.path.abspath(
    f'{DATA_PATH}/fine_tune_distillation_data'
)


class MNLI(torch.utils.data.Dataset):
    """Load MultiNLI dataset.

    Usage:
        train_data = MNLI('train')
        dev_matched = MNLI('dev_matched')
        dev_mismatched = MNLI('dev_mismatched')
    """

    @staticmethod
    def label_encoder(label):
        """Encode label into number.

        Label can only be 'entailment', 'neutral', 'contradiction',
        and will be encoded as 0, 1, 2, respectively.
        """

        if label == 'entailment':
            return 0
        elif label == 'neutral':
            return 1
        elif label == 'contradiction':
            return 2

        raise ValueError(f'unexpected label: {label}')

    @staticmethod
    def label_decoder(label_id):
        """Decode number into label.

        Number can only be 0, 1, 2, and will be dncoded as
        'entailment', 'neutral', 'contradiction', respectively.
        """

        if label_id == 0:
            return 'entailment'
        elif label_id == 1:
            return 'neutral'
        elif label_id == 2:
            return 'contradiction'

        raise ValueError(f'unexpected label id: {label_id}')

    @staticmethod
    def num_label():
        return 3

    @staticmethod
    def read_data(file_name):
        """Read MNLI data into structure.

        MNLI data must be downloaded previously,
        and put it in the path 'project_root/data/fine_tune_data/mnli/'.
        """

        mnli_file_path = f'{FINE_TUNE_DATA_PATH}/mnli/{file_name}.jsonl'
        with open(mnli_file_path) as mnli_jsonl_file:
            jsonlines = mnli_jsonl_file.read()

        data = []
        skip_data_counter = 0
        for jsonline in jsonlines.split('\n'):
            # Skip empty line.
            if jsonline == '':
                continue

            jsonobj = json.loads(jsonline)

            # Skip data which label is '-'.
            # See MNLI paper for more details.
            if jsonobj['gold_label'] == '-':
                skip_data_counter += 1
                continue

            # Format into transformer.tokenizer format.
            data.append({
                'text': jsonobj['sentence1'],
                'text_pair': jsonobj['sentence2'],
                'label': MNLI.label_encoder(jsonobj['gold_label']),
            })
        return data

    def __init__(self, file_name):
        self.data = MNLI.read_data(file_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def create_collate_fn(tokenizer):
        """Create `collate_fn` for `torch.utils.data.Dataloader`.

        Usage:
            dataset = MNLI('train')
            collate_fn = MNLI.create_collate_fn(tokenizer)
            data_loader = torch.utils.data.Dataloader(dataset,
                                                      collate_fn=collate_fn)
        """

        def collate_fn(batch):
            all_text_pair = [
                (obj['text'], obj['text_pair'])
                for obj in batch
            ]
            batch_encode = tokenizer.batch_encode_plus(all_text_pair,
                                                       pad_to_max_length=True,
                                                       return_tensors='pt')

            label = torch.LongTensor([obj['label'] for obj in batch])

            return (batch_encode['input_ids'],
                    batch_encode['attention_mask'],
                    batch_encode['token_type_ids'],
                    label)

        return collate_fn

    def add_soft_target(self, index, soft_target):
        """This function should only be used on 'train.jsonl'.

        Usage:
            index = 0
            dataset = MNLI('train')
            pred = model(dataset[index])
            dataset.add_soft_target(index=index,
                                    soft_target=pred)
        """
        self.data[index]['soft_target'] = soft_target

    def save_distillation_data(self, teacher_name, experiment_no):
        output_path = f'{FINE_TUNE_DISTILLATION_DATA_PATH}/mnli'
        output_file_path = f'{output_path}/{teacher_name}_experiment_{experiment_no}.jsonl'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(output_file_path, 'w') as mnli_jsonl_file:
            for index in range(len(self.data)):
                mnli_jsonl_file.write(f'{json.dumps(self.data[index])}\n')


class MNLIDistillation(torch.utils.data.Dataset):

    @staticmethod
    def read_data(teacher_name, experiment_no):
        """Read MNLI Distillation data into structure.

        MNLI Distillation data must be calculate previously,
        and put it in the path
        'project_root/data/fine_tune_distillation_data/mnli/'.
        """
        mnli_path = f'{FINE_TUNE_DISTILLATION_DATA_PATH}/mnli'
        mnli_file_path = f'{mnli_path}/{teacher_name}_experiment_{experiment_no}.jsonl'

        with open(mnli_file_path, 'r') as mnli_jsonl_file:
            jsonlines = mnli_jsonl_file.read()

        data = []
        for jsonline in jsonlines.split('\n'):
            # Skip empty line.
            if jsonline == '':
                continue

            data.append(json.loads(jsonline))

        return data

    def __init__(self, teacher_name, experiment_no):
        self.data = MNLIDistillation.read_data(
            teacher_name,
            experiment_no
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
