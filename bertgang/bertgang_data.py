from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle

from typing import List
from typing import Optional

import torch
import torch.utils
import torch.utils.data

from . import bertgang_util


class PreTrainData(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        ignore_files: Optional[List[str]] = None
    ) -> None:

        self.path = path
        self.all_files = {
            idx: file_name
            for idx, file_name in enumerate(bertgang_util.list_all_files(path, ignore_files=ignore_files))
        }

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        with open(f'{self.path}/{self.all_files[index]}', 'rb') as f:
            obj = pickle.load(f)

        obj['output_embeds'] = torch.FloatTensor(obj['output_embeds'])
        obj['token_type_ids'] = torch.LongTensor(obj['token_type_ids'])

        return obj
