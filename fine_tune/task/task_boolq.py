r""" BoolQ Pytorch dataset implementation
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os

from typing import Callable
from typing import List
from typing import Tuple
from typing import TypedDict

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import transformers

# my own modules

import fine_tune.path

###############################################################################
# Get logger.
###############################################################################

logger = logging.getLogger('fine_tune.task')

###############################################################################
# Define types for type annotation.
###############################################################################

class BoolQSample(TypedDict):
    r"""BoolQ Sample structure

    We structured sample as `transformers` model input.
    `soft_target` is only used when perform distillation,
    and it must remain `[0, 0, 0]` not used.
    """
    text: str
    text_pair: str
    label: int
    soft_target: List[float]

CollateFnReturn = Tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.LongTensor,
    torch.FloatTensor,
]

CollateFn = Callable[
    [
        List[BoolQSample],
    ],
    CollateFnReturn
]
###############################################################################
# Define BoolQ dataset and its utilities.
###############################################################################


class BoolQ(torch.utils.data.Dataset):
    r""" Load BoolQ dataset

    Usage:
    ---
        train_data = BoolQ('train')
        val_data = BoolQ('val')
        test_data = BoolQ('test')

    Args:
    ---
        dataset:
            BoolQ file to be loaded.
            `dataset` must be one of 'train', 'val' or 'test'.
            Otherwise `dataset` must be path to previous fine-tune
            experiment and contain files with name 'distill.jsonl'.

    Attributes:
    ---
        allow_dataset:
            Allowed BoolQ datasets.
            See BoolQ paper for more details.
        allow_labels:
            Allowed BoolQ labels, we do not consider '-' label.
            See BoolQ paper for more details.
        dataset:
            BoolQ dataset structured as `List[BoolQSample]`.
        task_path:
            Path of BoolQ dataset.
    """
    allow_dataset = (
        'train',
        'val',
        'test'
    )
    allow_labels = (
        False,
        True
    )

    task_path = f'{fine_tune.path.FINE_TUNE_DATA}/BoolQ'

    def __init__(self, dataset: str):
        if dataset in BoolQ.allow_dataset:
            logger.info('Start loading BoolQ %s', dataset)
            self.dataset = BoolQ.load(dataset)
            logger.info('Load BoolQ %s finished', dataset)
        else:
            logger.info('Start loading BoolQ distillation %s', dataset)
            self.dataset = BoolQ.load_distill(dataset)
            logger.info('Load BoolQ distillation %s finished', dataset)

    def __len__(self) -> int:
        r"""Return dataset size.

        Primary used by `torch.util.data.Dataloader`.

        Returns:
        ---
            An integer represent dataset size.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> BoolQSample:
        r"""Return sample by index
        Args:
        ---
            index:
                Sample index in BoolQ.
        Returns:
        ---
            A `dict` structured as `BoolQSample`.
        """
        return self.dataset[index]

    @staticmethod
    def label_encoder(label: str) -> int:
        r""" Encode label into number
        Args:
        ---
            label:
                `label` must be in one of the value of `allow_labels`.
                `label` is encoded as its respective index of `allow_labels`.
        Returns:
        ---
            Encode label represent by an integer
        """
        try:
            return BoolQ.allow_labels.index(label)
        except:
            raise ValueError(f'unexpected BoolQ label: {label}')

    @staticmethod
    def label_decoder(label_id: int) -> str:
        r"""Decode number into label.
        Args:
        ---
            label_id:
                `label_id` must satisfy the following condition:
                `0 <= label_id < len(BoolQ.allow_labels)`.
        Returns:
        ---
            Label which is decoded as `BoolQ.allow_labels[label_id]`.
        """
        try:
            # Avoid negative `label_id` like `label_id == -1`,
            # in order to abstract `allow_labels` from user.
            if not label_id >= 0:
                raise ValueError

            return BoolQ.allow_labels[label_id]
        except:
            raise ValueError(f"unexpected BoolQ label id: {label_id}")

    @staticmethod
    def num_class() -> int:
        r""" Return number of classes.

        Number of classes is corresponded to number of different labels.
        See `BoolQ.allow_labels` for label details.

        Returns:
        ---
            An integer represent number of classes.
        """
        return len(BoolQ.allow_labels)

    @staticmethod
    def load(dataset: str) -> List[BoolQSample]:
        r""" Load BoolQ dataset into memory.

        BoolQ dataset must be downloaded previously,
        and put it in the path 'project_root/data/fine_tune/mnli/'.
        See 'project_root/doc/fine_tune_boolq.md' for more information.

        Args:
        ---
            dataset:
                BoolQ file to be loaded.
                `dataset` must be one of 'train', 'val' or 'test'
        Returns:
        ---
            A list of samples structured as `BoolQSample`.
        Raises:
        ---
            FileNotFoundError:
                When BoolQ files does not exist.
        """

        try:
            file_path = f'{BoolQ.task_path}/{dataset}.jsonl'
            with open(file_path, 'r') as jsonl_file:
                jsonlines = jsonl_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'BoolQ dataset file {dataset} does not exist\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{file_path}\n' +
                "See '" +
                f'{fine_tune.path.DOC}/fine_tune_boolq.md' +
                "' for more information."
            )

        samples = []
        for line in jsonlines:
            # Skip empty line.
            if line == '':
                continue

            sample = json.loads(line)

            # Format into transformer.tokenizer format.
            # BoolQ labels will be encoded with `BoolQ.label_encoder`.
            samples.append(
                BoolQSample({
                    'text': sample['passage'],
                    'text_pair': sample['question'],
                    'label': BoolQ.label_encoder(sample['label']),
                    'soft_target':[0, 0]
                })
            )

        logger.info('Number of origin samples: %d', len(jsonlines))
        logger.info('Number of result samples: %d', len(samples))

        return samples

    @staticmethod
    def load_distill(formatted_experiment_name: str) -> List[BoolQSample]:
        r""" Load BoolQ distillation dataset into memory

        Warning:
        ---
            This method is under development
        """
        raise NotImplementedError(BoolQ.load_distill)

    @staticmethod
    def create_collate_fn(
            max_seq_len: int,
            tokenizer: transformers.PreTrainedTokenizer
    ) -> CollateFn:
        r"""Create `collate_fn` for `torch.utils.data.Dataloader`.
        Usage:
        ---
            dataset = BoolQ('train')
            collate_fn = BoolQ.create_collate_fn(tokenizer)
            data_loader = torch.utils.data.Dataloader(
                dataset,
                collate_fn=collate_fn
            )

        Args:
        ---
            max_seq_len:
                Max sequence length for fine-tune.
                When input sequence length smaller than `max_seq_len`, it will
                be padded to `max_seq_len`.
                When input sequence length bigger than `max_seq_len`, it will
                be truncated to `max_seq_len`.
            tokenizer:
                Pre-trained tokenizer provided by `transformers`.

        Retruns:
        ---
            A function used by `torch.utils.data.Dataloader`.
        """

        def collate_fn(batch: List[BoolQSample]) -> CollateFnReturn:
            text = []
            text_pair = []
            label = []
            soft_target = []

            for sample in batch:
                text.append(sample['text'])
                text_pair.append(sample['text_pair'])
                label.append(sample['label'])
                soft_target.append(sample['soft_target'])

            batch_encode = tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=max_seq_len,
                return_tensors='pt',
                truncation=True
            )

            return (
                batch_encode['input_ids'],
                batch_encode['attention_mask'],
                batch_encode['token_type_ids'],
                torch.LongTensor(label),
                torch.FloatTensor(soft_target)
            )
        return collate_fn

    def update_soft_target(self, index: int, soft_target: List[float]):
        r"""Update soft target by index
        Warning
        ---
            This method is under development
        """
        raise NotImplementedError(BoolQ.update_soft_target)

    def save_for_distillation(self, formatted_experiment_name: str): 
        r"""Save soft target for distillation.
        Warning
        ---
            This method is under development
        Args:
        ---
            formatted_experiment_name:
                Experiment name generated by `fine_tune.config.TeacherConfig`
        """
        raise NotImplementedError(BoolQ.save_for_distillation)
