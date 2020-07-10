"""All fine-tune dataset must go to this file.

Usage:
    dataset = MNLI('train')
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


class MNLISample(TypedDict):
    """MNLI Sample structure.

    We structured sample as `transformers` model input.
    `soft_target` is only used when perform distillation,
    and it must remain `[0, 0, 0]` not used.
    """
    text: str
    text_pair: str
    label: int
    soft_target: List[float]


CollateFnReturn = Tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
]

CollateFn = Callable[
    [
        List[MNLISample],
    ],
    CollateFnReturn
]

###############################################################################
# Define MNLI dataset and its utilities.
###############################################################################


class MNLI(torch.utils.data.Dataset):
    """Load MultiNLI dataset.

    Usage:
        train_data = MNLI('train')
        dev_matched = MNLI('dev_matched')
        dev_mismatched = MNLI('dev_mismatched')

    Args:
        dataset:
            MNLI file to be loaded.
            `dataset` must be one of 'train', 'dev_matched' or
            'dev_mismatched'.
            Otherwise `dataset` must be path to previous fine-tune
            experiment and contain files with name 'distill.jsonl'.

    Attributes:
        allow_dataset:
            Allowed MNLI datasets.
            See MNLI paper for more details.
        allow_labels:
            Allowed MNLI labels, we do not consider '-' label.
            See MNLI paper for more details.
        dataset:
            MNLI dataset structured as `List[MNLISample]`.
        task_path:
            Path of MNLI dataset.
    """

    allow_dataset = (
        'train',
        'dev_matched',
        'dev_mismatched',
    )

    allow_labels = (
        'entailment',
        'neutral',
        'contradiction',
    )

    task_path = f'{fine_tune.path.FINE_TUNE_DATA}/mnli'

    def __init__(self, dataset: str):
        if dataset in MNLI.allow_dataset:
            logger.info('Start loading MNLI %s', dataset)
            self.dataset = MNLI.load(dataset)
            logger.info('Load MNLI %s finished', dataset)
        else:
            logger.info('Start loading MNLI distillation %s', dataset)
            self.dataset = MNLI.load_distill(dataset)
            logger.info('Load MNLI distillation %s finished', dataset)

    def __len__(self) -> int:
        """Return dataset size.

        Primary used by `torch.util.data.Dataloader`.

        Returns:
            An integer represent dataset size.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> MNLISample:
        """Return sample by index.

        Args:
            index:
                Sample index in MNLI.

        Returns:
            A `dict` structured as `MNLISample`.
        """
        return self.dataset[index]

    @staticmethod
    def label_encoder(label: str) -> int:
        """Encode label into number.

        Args:
            label:
                `label` must be in one of the value of `allow_labels`.
                `label` is encoded as its respective index of `allow_labels`.

        Returns:
            Encoded label represent by an integer.
        """

        try:
            return MNLI.allow_labels.index(label)
        except:
            raise ValueError(f'unexpected MNLI label: {label}')

    @staticmethod
    def label_decoder(label_id: int) -> str:
        """Decode number into label.

        Args:
            label_id:
                `label_id` must satisfy the following condition:
                `0 <= label_id < len(MNLI.allow_labels)`.

        Returns:
            Label which is decoded as `MNLI.allow_labels[label_id]`.
        """

        try:
            # Avoid negative `label_id` like `label_id == -1`,
            # in order to abstract `allow_labels` from user.
            if not label_id >= 0:
                raise ValueError

            return MNLI.allow_labels[label_id]
        except:
            raise ValueError(f'unexpected MNLI label id: {label_id}')

    @staticmethod
    def num_class() -> int:
        """Return number of classes.

        Number of classes is corresponded to number of different labels.
        See `MNLI.allow_labels` for label details.

        Returns:
            An integer represent number of classes.
        """
        return len(MNLI.allow_labels)

    @staticmethod
    def load(dataset: str) -> List[MNLISample]:
        """Load MNLI dataset into memory.

        MNLI dataset must be downloaded previously,
        and put it in the path 'project_root/data/fine_tune/mnli/'.
        See 'project_root/doc/fine_tune_mnli.md' for more information.

        Args:
            dataset:
                MNLI file to be loaded.
                `dataset` must be one of 'train', 'dev_matched' or
                'dev_mismatched'.

        Returns:
            A list of samples structured as `MNLISample`.

        Raises:
            FileNotFoundError:
                When MNLI files does not exist.
        """

        try:
            file_path = f'{MNLI.task_path}/{dataset}.jsonl'
            with open(file_path, 'r') as jsonl_file:
                jsonlines = jsonl_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'MNLI dataset file {dataset} does not exist\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{file_path}\n' +
                "See '" +
                f'{fine_tune.path.DOC}/fine_tune_mnli.md' +
                "' for more information."
            )

        samples = []
        skipped_sample = 0
        for line in jsonlines:
            # Skip empty line.
            if line == '':
                continue

            sample = json.loads(line)

            # Skip sample which label is '-'.
            # See MNLI paper for more details.
            if sample['gold_label'] == '-':
                skipped_sample += 1
                continue

            # Format into transformer.tokenizer format.
            # MNLI labels will be encoded with `MNLI.label_encoder`.
            samples.append(
                MNLISample({
                    'text': sample['sentence1'],
                    'text_pair': sample['sentence2'],
                    'label': MNLI.label_encoder(sample['gold_label']),
                    'soft_target': [0, 0, 0],
                })
            )

        logger.info('Number of origin samples: %d', len(jsonlines))
        logger.info('Number of skiped samples: %d', skipped_sample)
        logger.info('Number of result samples: %d', len(samples))

        return samples

    @staticmethod
    def load_distill(formatted_experiment_name: str) -> List[MNLISample]:
        """Load MNLI distillation dataset into memory.

        MNLI distillation dataset must be saved previously,
        and put it in the path
        'project_root/data/fine_tune_experiment/some_experiment/distill.jsonl'.
        See 'project_root/doc/fine_tune_mnli.md' for more information.

        Args:
            formatted_experiment_name:
                Experiment name generated by
                `fine_tune.config.TeacherConfig`.

        Returns:
            A list of samples structured as `MNLISample`.

        Raises:
            FileNotFoundError:
                When MNLI files does not exist.
        """

        try:
            file_path = '{}/{}/distill.jsonl'.format(
                fine_tune.path.FINE_TUNE_EXPERIMENT,
                formatted_experiment_name
            )
            with open(file_path, 'r') as jsonl_file:
                jsonlines = jsonl_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                'MNLI distillation dataset file does not exist:\n' +
                formatted_experiment_name
            )

        samples = []
        for line in jsonlines:
            # Skip empty line.
            if line == '':
                continue

            samples.append(json.loads(line))

        logger.info('Number of distill samples: %d', len(samples))

        return samples

    @staticmethod
    def create_collate_fn(
            max_seq_len: int,
            tokenizer: transformers.PreTrainedTokenizer
    ) -> CollateFn:
        """Create `collate_fn` for `torch.utils.data.Dataloader`.

        Usage:
            dataset = MNLI('train')
            collate_fn = MNLI.create_collate_fn(tokenizer)
            data_loader = torch.utils.data.Dataloader(
                dataset,
                collate_fn=collate_fn
            )

        Args:
            max_seq_len:
                Max sequence length for fine-tune.
                When input sequence length smaller than `max_seq_len`, it will
                be padded to `max_seq_len`.
                When input sequence length bigger than `max_seq_len`, it will
                be truncated to `max_seq_len`.
            tokenizer:
                Pre-trained tokenizer provided by `transformers`.

        Returns:
            A function used by `torch.utils.data.Dataloader`.
        """

        def collate_fn(batch: List[MNLISample]) -> CollateFnReturn:
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
                torch.FloatTensor(soft_target),
            )

        return collate_fn

    def update_soft_target(
            self,
            index: int,
            soft_target: List[float],
    ):
        """Update soft target by index.

        This function should only be used on 'train.jsonl'.

        Usage:
            index = 0
            dataset = MNLI('train')
            dataset.update_soft_target(
                index=index,
                soft_target=list(model(dataset[index]))
            )

        Args:
            index:
                Sample index in MNLI.
            soft_target:
                Predict distribution from teacher model.
        """
        self.dataset[index]['soft_target'] = soft_target

    def save_for_distillation(
            self,
            formatted_experiment_name: str
    ):
        """Save soft target for distillation.

        Args:
            formatted_experiment_name:
                Experiment name generated by
                `fine_tune.config.TeacherConfig`.
        """
        output_dir = '{}/{}'.format(
            fine_tune.path.FINE_TUNE_EXPERIMENT,
            formatted_experiment_name
        )
        output_file_path = f'{output_dir}/distill.jsonl'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file_path, 'w') as jsonl_file:
            for index in range(len(self.dataset)):
                sample = json.dumps(self.dataset[index], ensure_ascii=False)
                jsonl_file.write(f'{sample}\n')
