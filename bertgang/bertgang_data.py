from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.utils
import torch.utils.data

from . import bertgang_config
from . import bertgang_tokenizer
from . import bertgang_util


class PreTrainData(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        config: bertgang_config.PreTrainConfig,
        tokenizer: bertgang_tokenizer.Tokenizer,
        ignore_files: Optional[List[str]] = None
    ) -> None:

        if not isinstance(config, bertgang_config.PreTrainConfig):
            raise TypeError(
                'parameter `config` must be type `bertgang_config.PreTrainConfig`.')

        self.config = config

        if not isinstance(tokenizer, bertgang_tokenizer.Tokenizer):
            raise TypeError(
                'parameter `tokenizer` must be type `bertgang_tokenizer.Tokenizer`.')

        self.tokenizer = tokenizer
        self.path = path

        all_files = bertgang_util.list_all_files(path, ignore_files=ignore_files)

        self.idx_file_mapping = {}

        for idx, file_name in enumerate(all_files):
            self.idx_file_mapping[idx] = file_name

    def __len__(self) -> int:
        return len(self.idx_file_mapping)

    def __getitem__(
        self,
        idx: int
    ) -> Dict[str, Union[str, torch.FloatTensor]]:
        target_dir = f'{self.path}/{self.idx_file_mapping[idx]}'

        with open(f'{target_dir}/original.pickle', 'rb') as f:
            input_obj = pickle.load(f)

        obj = self.tokenizer.encode_plus(
            text=input_obj['segment_a'],
            text_pair=input_obj['segment_b'],
            add_special_tokens=True,
            max_length=self.config.max_position_embeddings,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_tensors='pt',
            return_token_type_ids=True
        )

        for teacher in self.config.teachers:
            with open(f'{target_dir}/{teacher}.pickle', 'rb') as f:
                obj[f'{teacher}_output_embeds'] = torch.FloatTensor(pickle.load(f))

        return obj

    @staticmethod
    def collate_fn(
        obj_list: List[Dict]
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for obj in obj_list:
            input_ids.append(obj['input_ids'].unsqueeze(0))
            token_type_ids.append(obj['token_type_ids'].unsqueeze(0))
            attention_mask.append(obj['attention_mask'].unsqueeze(0))

        # TODO: return teacher tensor
        return (torch.cat(input_ids),
                torch.cat(token_type_ids),
                torch.cat(attention_mask),)

    def get_data_loader(self) -> None:
        return torch.utils.data.DataLoader(self,
                                           batch_size=self.config.batch_size,
                                           collate_fn=self.__class__.collate_fn,
                                           drop_last=False,
                                           shuffle=True)
