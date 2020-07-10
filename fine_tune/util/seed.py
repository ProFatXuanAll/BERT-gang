"""Helper function for setting random seed.

Usage:
    fine_tune.util.set_seed(...)
    fine_tune.util.set_seed_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

# 3rd party modules

import numpy as np
import torch

# my own modules

import fine_tune.config
import fine_tune.model
import fine_tune.task


def set_seed(num_gpu: int, seed: int):
    """Set random seed for experiment reproducibility.

    Args:
        num_gpu:
            Number of GPUs used to run experiment.
        seed:
            An integer stands for random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and num_gpu >= 1:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed_by_config(
        config: fine_tune.config.TeacherConfig
):
    """Set random seed for experiment reproducibility.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes `seed`
            and `num_gpu`.
    """
    set_seed(
        num_gpu=config.num_gpu,
        seed=config.seed
    )
