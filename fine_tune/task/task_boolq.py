r""" BoolQ Pytorch dataset implementation
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os

from typing import Callable
from typing import List
from typing import Tuple
from typing import TypedDict

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import transformers

# my own modules

import fine_tune.path

###############################################################################
# Get logger.
###############################################################################

logger = logging.getLogger('fine_tune.task')

###############################################################################
# Define types for type annotation.
###############################################################################

class BoolQSample(TypedDict):
    r"""BoolQ Sample structure

    We structured sample as `transformers` model input.
    `soft_target` is only used when perform distillation,
    and it must remain `[0, 0, 0]` not used.
    """
    text: str
    text_pair: str
    label: int
    soft_target: List[float]

CollateFnReturn = Tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.LongTensor,
    torch.FloatTensor,
]

CollateFn = Callable[
    [
        List[BoolQSample],
    ],
    CollateFnReturn
]
###############################################################################
# Define BoolQ dataset and its utilities.
###############################################################################