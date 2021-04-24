r"""Implement a custom batch sampler for contrastive learning objective.
Why we need this sampler during training process ?
Cause in `fine_tune.contrast_util.SupConLoss`, we may encounter an `NaN` bug,
which is caused by the imbalanced number of labels in each mini-batch.
As a result, we need a custom batch sampler to guarantee
the balanced number of labels in each mini-batch.
"""
# Built-in modules.
import warnings

# 3rd-party modules.

import numpy as np

# My own modules.

from fine_tune.task._dataset import Dataset
from fine_tune.task._mnli import MNLI


class GlueBatchSampler():
    r"""Class `GlueBatchSampler` will sample given GLUE dataset randomly
    (i.e: shuffle order) and guarantee number of labels in each mini-batch
    is greater than 1
    For example, `MNLI` task with `batch_size` == 8:

    Examples
    ----------
    >>> dataset = fine_tune.task.MNLI('train')
    >>> dataloader = torch.utils.data.DataLoader(
    ...     dataset,
    ...     batch_sampler=GlueBatchSampler(dataset, 4),
    ...     collate_fn=dataset.create_collate_fn())
    >>> for text, text_pari, labels in dataloader:
    ...     print(labels)
    [0,1,2,0,1,2,0,1]

    Notes
    ----------
    1.This batch sampler should just be used during training.
    2.Please make sure to use the function in `fine_tune.util.seed`
    to control environment random seed.

    Parameters
    ----------
    datasets : fine_tune.task.Dataset
        GLUE training dataset.
    batch_size : int
        Batch size.
    drop_last : bool, optional
        If `True`, the sampler will drop the last batch if
        its size would be less than `batch_size`
        Default, False
    """

    def __init__(self, datasets: Dataset, batch_size: int, drop_last: bool = False):
        # Validate datasets.
        if not isinstance(datasets, Dataset):
            raise ValueError("Invalid custom dataset!")

        if batch_size % 2 != 0:
            warnings.warn(
                "It is recommended to use an even batch size\n"+
                f"Current batch size: {batch_size}"
            )

        if batch_size <= 6:
            raise ValueError(
                "In order to prevent `NaN` runtime error "+
                "batch size should be greater than 6\n"+
                f"Current batch size: {batch_size}"
            )

        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Label indices look up table.
        self.label_indices_table = {}
        for idx, sample in enumerate(self.datasets):
            if sample['label'] in self.label_indices_table:
                self.label_indices_table[sample['label']].append(idx)
            else:
                self.label_indices_table.update({sample['label']:[idx]})

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        batch = []

        if isinstance(self.datasets, MNLI):
            counter = 0
            for _ in range(len(self.datasets)):
                if counter < 3:
                    batch.append(
                        int(np.random.choice(
                            self.label_indices_table[counter],
                            1,
                            replace = False
                        ))
                    )
                else:
                    raise ValueError(f"Incorrect counter: {str(counter)}")

                counter += 1
                if counter % 3 == 0:
                    counter = 0
                if len(batch) == self.batch_size:
                    np.random.shuffle(batch)
                    yield batch
                    batch = []

            if len(batch) > 0 and not self.drop_last:
                yield batch
        else:
            counter = 0
            for _ in range(len(self.datasets)):
                if counter < 2:
                    batch.append(
                        int(np.random.choice(
                            self.label_indices_table[counter],
                            1,
                            replace = False
                        ))
                    )
                else:
                    raise ValueError(f"Incorrect counter: {str(counter)}")
                counter += 1
                if counter % 2 == 0:
                    counter = 0
                if len(batch) == self.batch_size:
                    np.random.shuffle(batch)
                    yield batch
                    batch = []

            if len(batch) > 0 and not self.drop_last:
                yield batch
