r"""Helper function for loading optimizer.

Usage:
    optimizer = fine_tune.util.load_optimizer(...)
    optimizer = fine_tune.util.load_optimizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Tuple
from typing import Union

# 3rd party modules

import torch
import torch.optim

# my own modules

import fine_tune.config
import fine_tune.model
import fine_tune.task


def load_optimizer(
        betas: Tuple[float, float],
        eps: float,
        learning_rate: float,
        model: Union[
            fine_tune.model.StudentAlbert,
            fine_tune.model.StudentBert,
            fine_tune.model.TeacherAlbert,
            fine_tune.model.TeacherBert,
        ],
        weight_decay: float
) -> torch.optim.AdamW:
    r"""Load AdamW optimizer.

    Args:
        betas:
            Optimizer AdamW's parameter `betas`.
        eps:
            Optimizer AdamW's parameter `eps`.
        learning_rate:
            Optimizer AdamW's parameter `lr`.
        model:
            Source parameters to be optimized.
        weight_decay:
            Optimizer AdamW's parameter `weight_decay`.

    Returns:
        AdamW optimizer.
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    return optimizer


def load_optimizer_by_config(
        config: Union[
            fine_tune.config.StudentConfig,
            fine_tune.config.TeacherConfig,
        ],
        model: Union[
            fine_tune.model.StudentAlbert,
            fine_tune.model.StudentBert,
            fine_tune.model.TeacherAlbert,
            fine_tune.model.TeacherBert,
        ]
) -> torch.optim.AdamW:
    r"""Load AdamW optimizer.

    Args:
        config:
            Configuration object which contains attributes
            `learning_rate`, `betas`, `eps` and `weight_decay`.
        model:
            Source parameters to be optimized.

    Returns:
        Same as `fine_tune.util.load_optimizer`.
    """

    return load_optimizer(
        betas=config.betas,
        eps=config.eps,
        learning_rate=config.learning_rate,
        model=model,
        weight_decay=config.weight_decay
    )
