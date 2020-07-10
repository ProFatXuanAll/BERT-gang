"""Path constant shared by all files."""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# project folder path
PROJECT_ROOT = os.path.abspath(
    f'{os.path.abspath(__file__)}/../..'
)

# data folder
DATA = os.path.abspath(
    f'{PROJECT_ROOT}/data'
)

if not os.path.exists(DATA):
    os.makedirs(DATA)

# doc folder
DOC = os.path.abspath(
    f'{PROJECT_ROOT}/doc'
)

# fine tune data folder
FINE_TUNE_DATA = os.path.abspath(
    f'{DATA}/fine_tune'
)

if not os.path.exists(FINE_TUNE_DATA):
    os.makedirs(FINE_TUNE_DATA)

# fine tune experiment folder
FINE_TUNE_EXPERIMENT = os.path.abspath(
    f'{DATA}/fine_tune_experiment'
)

if not os.path.exists(FINE_TUNE_EXPERIMENT):
    os.makedirs(FINE_TUNE_EXPERIMENT)

# log folder for fine tune experiment
LOG = os.path.abspath(
    f'{FINE_TUNE_EXPERIMENT}/log'
)

if not os.path.exists(LOG):
    os.makedirs(LOG)

# configuration file name
CONFIG = 'config.json'
