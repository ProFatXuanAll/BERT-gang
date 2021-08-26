r"""QNLI dataset.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.QNLI('train')
    dataset = fine_tune.task.QNLI('dev')

    dataloader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=QNLI.create_collate_fn(...)
    )
"""

# Built-in modules.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from typing import List

# 3rd party modules

from tqdm import tqdm

# my own modules

import fine_tune.path

from fine_tune.task._dataset import (
    Dataset,
    Label,
    Sample,
    label_encoder
)

# Get logger.

logger = logging.getLogger('fine_tune.task')

# Define QNLI dataset.

class QNLI(Dataset):
    r"""QNLI dataset and its utilities

    Parameters
    ----------
    Dataset : string
        Name of QNLI dataset file to be loaded.
    """
    allow_dataset: List[str] = [
        'train',
        'dev',
        'test'
    ]

    allow_labels: List[Label] = [
        'not_entailment',
        'entailment'
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'QNLI'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        r"""Load QNLI dataset into memory.

        This is a heavy IO method and might required lots of memory since
        dataset might be huge. QNLI dataset must be download previously. See
        QNLI document in 'project_root/doc/fine_tune_qnli.md' for downloading
        details.
        Parameters
        ----------
        dataset : str
            Name of the QNLI dataset to be loaded.
        Returns
        -------
        List[Sample]
            A list of QNLI samples.
        """
        try:
            dataset_path = os.path.join(
                QNLI.task_path,
                f'{dataset}.tsv'
            )
            if not 'test' in dataset:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading QNLI {dataset}'):
                        sample = sample.strip()
                        index, question, sentence, label = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': question,
                                'text_pair': sentence,
                                'label': label_encoder(QNLI, label)
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
            else:
                with open(dataset_path, 'r') as tsv_file:
                    # Skip first line.
                    tsv_file.readline()
                    samples = []
                    for sample in tqdm(tsv_file, desc=f'Loading QNLI {dataset}'):
                        sample = sample.strip()
                        index, question, sentence = sample.split('\t')
                        samples.append(
                            Sample({
                                'index': int(index),
                                'text': question,
                                'text_pair': sentence,
                                'label': -1
                            })
                        )

                    logger.info(
                        'Number of samples: %d',
                        len(samples)
                    )

                return samples
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'QNLI dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_qnli.md') +
                "' for downloading details."
            ) from error
