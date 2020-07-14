r"""Helper function for loading dataset.

Usage:
    dataset = fine_tune.util.load_dataset(...)
    dataset = fine_tune.util.load_dataset_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# my own modules

import fine_tune.config
import fine_tune.task


def load_dataset(
        dataset: str,
        task: str,
) -> Union[
    fine_tune.task.MNLI,
    fine_tune.task.BoolQ
]:
    r"""Load task specific dataset.

    Args:
        dataset:
            Dataset name of particular fine-tune task.
        task:
            Name of fine-tune task.

    Returns:
        `fine_tune.task.MNLI`:
            If `config.task` is 'mnli'.
        `fine_tune.task.BoolQ`:
            If `config.task` is 'boolq'/
    """
    if task == 'mnli':
        return fine_tune.task.MNLI(dataset)
    elif task == 'boolq':
        return fine_tune.task.BoolQ(dataset)

def load_dataset_by_config(
        config: Union[
            fine_tune.config.StudentConfig,
            fine_tune.config.TeacherConfig,
        ]
) -> Union[
    fine_tune.task.MNLI,
]:
    r"""Load task specific dataset.

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
