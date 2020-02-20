from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import random

from typing import List
from typing import Optional

import torch
import transformers


def list_all_files(path: str,
                   ignore_files: Optional[List[str]] = None
                   ) -> List[str]:
    """List all files in a directory.

    Args:
        path (str):
            Path of a directory which containing pretrain data.
        ignore_files (list[str]):
            Data file names to be not included in the return list.
            Default to None.
    Raises:
        TypeError:
            If `path` is not type `str` or `ignore_files` is not type
            `list(str)`.
        FileNotFoundError:
            If `path` does not exist.
            If files list in `ignore_files` do not exist, then no exception
            will be raised since they were meant to be ignored.
        OSError:
            If `path` is not a directory.
    Returns:
        list[str]:
            Data file names in the given path, sorted by dictionary order.
    """
    # check parameter
    if not isinstance(path, str):
        raise TypeError('parameter `path` must be type `str`.')

    if not os.path.exists(path):
        raise FileNotFoundError(f'directory `{path}` does not exist.')

    if not os.path.isdir(path):
        raise OSError(f'{path} is not a directory.')

    if ignore_files is None:
        ignore_files = []
    else:
        if not isinstance(ignore_files, list):
            raise TypeError(
                'parameter `ignore_files` must be type `list[str]`.')

        for ignore_file in ignore_files:
            if not isinstance(ignore_file, str):
                raise TypeError(
                    'parameter `ignore_files` must be type `list[str]`.')

    all_files = os.listdir(path)
    all_files = sorted(all_files)

    for ignore_file in ignore_files:
        if ignore_file in all_files:
            all_files.remove(ignore_file)

    return all_files
