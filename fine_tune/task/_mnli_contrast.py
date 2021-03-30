r"""MNLI dataset for contrastive learning.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.MNLIContrast('train', k=10, defined_by_label=False)
    dataset = fine_tune.task.MNLIContrast('dev_matched', k=10, defined_by_label=False)
    dataset = fine_tune.task.MNLIContrast('dev_mismatched', k=10, defined_by_label=False)
    dataset = fine_tune.task.MNLIContrast(...)

    assert fine_tune.task.get_num_label(fine_tune.task.MNLIContrast) == 3

    assert fine_tune.task.label_encoder(
        fine_tune.task.MNLIContrast,
        fine_tune.task.MNLIContrast.allow_labels[0]
    ) == 0

    assert fine_tune.task.label_decoder(
        fine_tune.task.MNLIContrast,
        0
    ) == fine_tune.task.MNLIContrast.allow_labels[0]

    data_loader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=MNLIContrast.create_collate_fn(...)
    )
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os

from typing import List

# 3rd party modules

from tqdm import tqdm

# my own modules

import fine_tune.path

from fine_tune.task._dataset import (
    ContrastDataset,
    Label,
    Sample,
    label_encoder
)

# Get logger.

logger = logging.getLogger('fine_tune.task')

# Define MNLIContrast dataset.

class MNLIContrast(ContrastDataset):
    r"""MultiNLI dataset for contrastive learning and its utilities.

    Args:
    ---
        dataset:
            Name of MNLI dataset file to be loaded.
        k:
            Number of negative samples.
        defined_by_label:
            Use label information to define positive and negative sample.

    Attributes:
    ---
        allow_dataset:
            Allowed MNLI dataset. See MNLI paper for more details.
        allow_labels:
            Allowed MNLI labels. We do not consider '-' label. See MNLI paper
            for labeling details.
        dataset:
            A list of MNLI samples.
        task_path:
            Path of MNLI dataset.
    """
    allow_dataset: List[str] = [
        'train',
        'dev_matched',
        'dev_mismatched',
    ]

    allow_labels: List[Label] = [
        'entailment',
        'neutral',
        'contradiction',
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'mnli'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        """Load MNLI dataset into memory.

        This is a heavy IO method and might required lots of memory since
        dataset might be huge. MNLI dataset must be download previously. See
        MNLI document in 'project_root/doc/fine_tune_mnli.md' for downloading
        details.

        Parameters
        ----------
        datset : str
            Name of the MNLI dataset to be loaded.

        Returns
        -------
        List[Sample]
            A list of MNLI samples.

        Raises
        ------
            FileNotFoundError:
                When MNLI files does not exist.
        """
        try:
            dataset_path = os.path.join(
                MNLIContrast.task_path,
                f'{dataset}.jsonl'
            )
            with open(dataset_path, 'r') as jsonl_file:
                jsonlines = jsonl_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'MNLI dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_mnli.md') +
                "' for downloading details."
            )

        samples = []
        skipped_sample_count = 0
        for line in tqdm(jsonlines, desc=f'Loading MNLI {dataset}'):
            # Skip empty line.
            if line == '':
                continue

            sample = json.loads(line)

            # Skip sample which label is '-'. See MNLI paper for labeling
            # details.
            if sample['gold_label'] == '-':
                skipped_sample_count += 1
                continue

            # Format into `transformer.PreTrainedTokenizer` format. MNLI labels
            # will be encoded with `label_encoder(MNLI, label)`. `logits`
            # will be initialized with 3 zeros.
            samples.append(
                Sample({
                    'text': sample['sentence1'],
                    'text_pair': sample['sentence2'],
                    'label': label_encoder(MNLIContrast, sample['gold_label'])
                })
            )

        logger.info(
            'Number of origin samples: %d',
            len(samples) + skipped_sample_count
        )
        logger.info('Number of skiped samples: %d', skipped_sample_count)
        logger.info('Number of result samples: %d', len(samples))

        return samples
