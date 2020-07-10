r"""Helper function for loading tokenizer.

Usage:
    tokenizer = fine_tune.util.load_tokenizer(...)
    tokenizer = fine_tune.util.load_tokenizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# 3rd party modules

import transformers

# my own modules

import fine_tune.config
import fine_tune.model
import fine_tune.task


def load_tokenizer(
        pretrained_version: str,
        teacher: str
) -> Union[
    transformers.AlbertTokenizer,
    transformers.BertTokenizer,
]:
    r"""Load tokenizer.

    Args:
        pretrained_version:
            Pretrained model provided by hugginface.
        teacher:
            Teacher model's name.

    Returns:
        `transformers.AlbertTokenizer`:
            If `teacher` is 'albert'.
        `transformers.BertTokenizer`:
            If `teacher` is 'bert'.
    """

    if teacher == 'albert':
        return transformers.AlbertTokenizer.from_pretrained(
            pretrained_version
        )
    if teacher == 'bert':
        return transformers.BertTokenizer.from_pretrained(
            pretrained_version
        )


def load_tokenizer_by_config(
        config: fine_tune.config.TeacherConfig
) -> Union[
    transformers.AlbertTokenizer,
    transformers.BertTokenizer,
]:
    r"""Load tokenizer.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `pretrained_version` and `teacher`.

    Returns:
        Same as `fine_tune.util.load_tokenizer`.
    """
    return load_tokenizer(
        pretrained_version=config.pretrained_version,
        teacher=config.teacher
    )
