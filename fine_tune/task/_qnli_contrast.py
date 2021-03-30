r"""QNLI dataset for contrastive learning.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.QNLIContrast('train', k=10, defined_by_label=False)
    dataset = fine_tune.task.QNLIContrast('dev_matched', k=10, defined_by_label=False)
    dataset = fine_tune.task.QNLIContrast('dev_mismatched', k=10, defined_by_label=False)
    dataset = fine_tune.task.QNLIContrast(...)

    assert fine_tune.task.get_num_label(fine_tune.task.QNLIContrast) == 3

    assert fine_tune.task.label_encoder(
        fine_tune.task.QNLIContrast,
        fine_tune.task.QNLIContrast.allow_labels[0]
    ) == 0

    assert fine_tune.task.label_decoder(
        fine_tune.task.QNLIContrast,
        0
    ) == fine_tune.task.QNLIContrast.allow_labels[0]

    data_loader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=QNLIContrast.create_collate_fn(...)
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

# Define QNLIContrast dataset.

class QNLIContrast(ContrastDataset):
    r"""QNLI dataset for contrastive learning and its utilities

    Parameters
    ----------
        dataset : string
            Name of QNLI dataset file to be loaded.
        k:
            Number of negative samples.
        defined_by_label:
            Use label information to define positive and negative sample.
    """
    # TODO: support testing set.
    allow_dataset: List[str] = [
        'train',
        'dev'
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
                QNLIContrast.task_path,
                f'{dataset}.tsv'
            )
            with open(dataset_path, 'r') as tsv_file:
                # Skip firt line.
                tsv_file.readline()
                samples = []
                for sample in tqdm(tsv_file, desc=f'Loading QNLI {dataset}'):
                    sample = sample.strip()
                    _, question, sentence, label = sample.split('\t')
                    samples.append(
                        Sample({
                            'text': question,
                            'text_pair': sentence,
                            'label': label_encoder(QNLIContrast, label)
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
