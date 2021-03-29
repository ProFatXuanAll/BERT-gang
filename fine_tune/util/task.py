r"""Helper functions for loading dataset.

Usage:
    import fine_tune

    dataset = fine_tune.util.load_dataset(...)
    dataset = fine_tune.util.load_dataset_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

import fine_tune.config
import fine_tune.task


def load_dataset(
        dataset: str,
        task: str,
) -> fine_tune.task.Dataset:
    r"""Load fine-tune task's dataset.

    Args:
        dataset:
            Name of the datset file to be loaded. When `dataset` is the name of
            some previous experiment, it means the logits dataset generated by
            the model of that experiment.
        task:
            Name of the fine-tune task.

    Raises:
        ValueError:
            If `task` does not supported.

    Returns:
        `fine_tune.task.MNLI`:
            If `task` is 'mnli'.
        `fine_tune.task.BoolQ`:
            If `task` is 'boolq'.
    """
    if task == 'mnli':
        return fine_tune.task.MNLI(dataset)
    if task == 'boolq':
        return fine_tune.task.BoolQ(dataset)
    if task == 'qnli':
        return fine_tune.task.QNLI(dataset)

    raise ValueError(
        f'`task` {task} is not supported.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--task {option}',
            [
                'mnli',
                'boolq'
            ]
        )))
    )


def load_dataset_by_config(
        config: fine_tune.config.BaseConfig
) -> fine_tune.task.Dataset:
    r"""Load fine-tune task's dataset.

    Args:
        config:
            Configuration object which contains attributes `task`
            and `dataset`.

    Returns:
        Same as `fine_tune.util.load_data`.
    """
    return load_dataset(
        dataset=config.dataset,
        task=config.task
    )
