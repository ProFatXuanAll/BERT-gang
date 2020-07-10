r"""Helper function for loading model.

Usage:
    model = fine_tune.util.load_teacher_model(...)
    model = fine_tune.util.load_teacher_model_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# 3rd party modules

import torch

# my own modules

import fine_tune.config
import fine_tune.model


def load_teacher_model(
        device: torch.device,
        dropout: float,
        num_class: int,
        num_gpu: int,
        pretrained_version: str,
        teacher: str,
) -> Union[
    fine_tune.model.TeacherAlbert,
    fine_tune.model.TeacherBert,
]:
    r"""Load teacher model.

    Args:
        device:
            CUDA device to run model.
        dropout:
            Dropout probability.
        num_class:
            Number of classes to classify.
        num_gpu:
            Number of GPUs to train.
        pretrained_version:
            Pretrained model provided by hugginface.
        teacher:
            Teacher model's name.

    Returns:
        `fine_tune.model.TeacherAlbert`:
            If `config.teacher` is 'albert'.
        `fine_tune.model.TeacherBert`:
            If `config.teacher` is 'bert'.
    """
    # if config.num_gpu >= 1:
    #     # Make sure only the first process in distributed training
    #     # will download model & vocab.
    #     torch.distributed.barrier()

    if teacher == 'albert':
        model = fine_tune.model.TeacherAlbert(
            dropout=dropout,
            num_class=num_class,
            pretrained_version=pretrained_version
        )
    if teacher == 'bert':
        model = fine_tune.model.TeacherBert(
            dropout=dropout,
            num_class=num_class,
            pretrained_version=pretrained_version
        )

    if num_gpu >= 1:
        model = model.to(device)

    return model


def load_teacher_model_by_config(
        config: fine_tune.config.TeacherConfig
) -> Union[
    fine_tune.model.TeacherAlbert,
    fine_tune.model.TeacherBert,
]:
    r"""Load teacher model.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `dropout`, `pretrained_version` and `num_class`.

    Returns:
        Same as `fine_tune.util.load_teacher_model`.
    """
    return load_teacher_model(
        device=config.device,
        dropout=config.dropout,
        num_class=config.num_class,
        num_gpu=config.num_gpu,
        pretrained_version=config.pretrained_version,
        teacher=config.teacher
    )
