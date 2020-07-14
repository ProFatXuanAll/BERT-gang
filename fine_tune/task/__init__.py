r"""All fine-tune tasks.

All fine-tune tasks files in this module must begin with `task_`
and renamed in this very file.
This help to avoid unnecessary import structure (we prefer using
`fine_tune.task.MNLI` over `fine_tune.task.task_mnli.MNLI`).

Usage:
    dataset = fine_tune.task.MNLI(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

from fine_tune.task.task_mnli import MNLI
from fine_tune.task.task_boolq import BoolQ
