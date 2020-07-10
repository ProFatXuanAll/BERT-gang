r"""Helper function for loading scheduler.

Usage:
    scheduler = fine_tune.util.load_scheduler(...)
    scheduler = fine_tune.util.load_scheduler_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# 3rd party modules

import numpy as np
import torch
import torch.optim
import transformers

# my own modules

import fine_tune.config
import fine_tune.model
import fine_tune.task


def load_scheduler(
        batch_size: int,
        dataset: Union[
            fine_tune.task.MNLI,
        ],
        epoch: int,
        optimizer: torch.optim.AdamW,
        warmup_step: int
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Load linear warmup scheduler.

    Args:
        dataset:
            Fine-tune task specific dataset.
            Used to calculate total training step.
        optimizer:
            AdamW optimizer.

    Returns:
        Linear warmup scheduler implemented by hugginface.
    """
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=int(
            np.ceil(len(dataset) / batch_size)
        ) * epoch
    )

    return scheduler


def load_scheduler_by_config(
        config: fine_tune.config.TeacherConfig,
        dataset: Union[
            fine_tune.task.MNLI,
        ],
        optimizer: torch.optim.AdamW,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Load linear warmup scheduler.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `batch_size`, `epoch` and `warmup_step`.
        dataset:
            Fine-tune task specific dataset.
            Used to calculate total training step.
        optimizer:
            AdamW optimizer.

    Returns:
        Same as `fine_tune.util.load_scheduler`.
    """
    return load_scheduler(
        batch_size=config.batch_size,
        dataset=dataset,
        epoch=config.epoch,
        optimizer=optimizer,
        warmup_step=config.warmup_step
    )
