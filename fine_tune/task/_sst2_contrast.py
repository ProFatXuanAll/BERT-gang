r"""SST2 dataset for contrastive learning.

Usage:
    import torch.utils.data.Dataloader
    import fine_tune

    dataset = fine_tune.task.SST2Contrast('train', k=10, defined_by_label=False)
    dataset = fine_tune.task.SST2Contrast('dev', k=10, defined_by_label=False)
    dataset = fine_tune.task.SST2Contrast(...)

    assert fine_tune.task.get_num_label(fine_tune.task.SST2Contrast) == 2

    assert fine_tune.task.label_encoder(
        fine_tune.task.SST2Contrast,
        fine_tune.task.SST2Contrast.allow_labels[0]
    ) == 0

    assert fine_tune.task.label_decoder(
        fine_tune.task.SST2Contrast,
        0
    ) == fine_tune.task.SST2Contrast.allow_labels[0]

    data_loader = torch.utils.data.Dataloader(
        dataset,
        collate_fn=SST2Contrast.create_collate_fn(...)
    )
"""

# built-in modules

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
    ContrastDataset,
    Label,
    Sample,
    label_encoder
)

# Get logger.

logger = logging.getLogger('fine_tune.task')

# Define SST2Contrast dataset.

class SST2Contrast(ContrastDataset):
    r"""SST2 dataset for contrastive learning and its utilities

    Parameters
    ----------
        dataset : string
            Name of SST2 dataset file to be loaded.
        k:
            Number of negative samples.
        defined_by_label:
            Use label information to define positive and negative sample.
    """
    #TODO: support testing dataset.
    allow_dataset: List[str] = [
        'train',
        'dev'
    ]

    allow_labels: List[Label] = [
        '0',
        '1'
    ]

    task_path: str = os.path.join(
        fine_tune.path.FINE_TUNE_DATA,
        'SST-2'
    )

    @staticmethod
    def load(dataset: str) -> List[Sample]:
        r"""Load SST2 dataset into memory.

        This is a heavy IO method and might required lots of memory since
        dataset might be huge. SST2 dataset must be download previously. See
        SST2 document in 'project_root/doc/fine_tune_sst2.md' for downloading
        details.
        Parameters
        ----------
        dataset : str
            Name of the SST2 dataset to be loaded.
        Returns
        -------
        List[Sample]
            A list of SST2 samples.
        """
        try:
            dataset_path = os.path.join(
                SST2Contrast.task_path,
                f'{dataset}.tsv'
            )
            with open(dataset_path, 'r') as tsv_file:
                # Skip first line.
                tsv_file.readline()
                samples = []
                for sample in tqdm(tsv_file, desc=f'Loading SST2 {dataset}'):
                    sample = sample.strip()
                    sentence, label = sample.split('\t')
                    samples.append(
                        Sample({
                            'text' : sentence,
                            'text_pair' : None,
                            'label': label_encoder(SST2Contrast, label)
                        })
                    )

                logger.info(
                    'Number of samples: %d',
                    len(samples)
                )

            return samples
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'SST2 dataset file {dataset} does not exist.\n' +
                'You must downloaded previously and put it in the path:\n' +
                f'{dataset_path}\n' +
                "See '" +
                os.path.join(fine_tune.path.DOC, 'fine_tune_sst2.md') +
                "' for downloading details."
            ) from error
