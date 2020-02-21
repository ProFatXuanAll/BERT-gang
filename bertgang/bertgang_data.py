"""Data classes for pre-train and fine-tune experiments.

Use `bertgang.bertgang_data.PreTrainData` to create dataset for pre-train
experiments.
Use `bertgang.bertgang_data.FineTuneData` to create dataset for fine-tune
experiments.

TODO:
    - The following method should return teacher tensor:
        - `bertgang.bertgang_data.PreTrainData.__getitem__`
        - `bertgang.bertgang_data.PreTrainData.collate_fn`
    - Add `bertgang.bertgang_data.FineTuneData` class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
    """Dataset class for pre-train experiments.

    Args:
        path (str):
            Directory path containing pre-train data.
            `path` must be the following structure:

            - pre-train-data-1 (a directory)
                - original.pickle (a pickle file)
                - teacher-1-name.pickle (a pickle file)
                - teacher-2-name.pickle (a pickle file)
                - teacher-n-name.pickle (a pickle file)
            - pre-train-data-2 (same structure as above)
            - pre-train-data-m (same structure as above)
        config (bertgang.bertgang_config.PreTrainConfig):
            Source of teachers list.
        tokenizer (bertgang.bertgang_tokenizer.Tokenizer):
            Pre-trained tokenizer.
        ignore_files (list(str)):
            Directories need to be ignored from `path`.
            Default to None.
        is_checking_files (bool):
            Whether to check all pre-train files.
            Since there are so many pre-train files, it will be time-consuming
            to check each pre-train data.
            We suggest that only check pre-train data once and ignore checking
            for the rest processes.
            Default to False.

    Attributes:
        path (str):
            Same as parameter `path`.
            Mainly used by the methods
            `bertgang.bertgang_data.PreTrainData.__getitem__`.
        config (bertgang.bertgang_config.PreTrainConfig):
            Same as parameter `config`.
            Mainly used by the following methods:

            - `bertgang.bertgang_data.PreTrainData.__init__`
            - `bertgang.bertgang_data.PreTrainData.__getitem__`
            - `bertgang.bertgang_data.PreTrainData.get_data_loader`
        tokenizer (bertgang.bertgang_tokenizer.Tokenizer):
            Same as parameter `tokenizer`.
            Mainly used by the methods
            `bertgang.bertgang_data.PreTrainData.__getitem__`.
        idx_dir_mapping (dict):
            Map each pre-train data by an unique integer.
            Mainly used by the following methods:

            - `bertgang.bertgang_data.PreTrainData.__len__`
            - `bertgang.bertgang_data.PreTrainData.__getitem__`

    Raises:
        TypeError:
            If `config` is not type
            `bertgang.bertgang_config.PreTrainConfig`,
            or `tokenizer` is not type
            `bertgang.bertgang_tokenizer.Tokenizer`.
    """

    def __init__(
            self,
            path: str,
            config: bertgang_config.PreTrainConfig,
            tokenizer: bertgang_tokenizer.Tokenizer,
            ignore_files: Optional[List[str]] = None,
            is_checking_files: bool = False
    ) -> None:

        # check parameters
        if not isinstance(config, bertgang_config.PreTrainConfig):
            raise TypeError(
                'parameter `config` must be type '
                '`bertgang.bertgang_config.PreTrainConfig`.'
            )

        self.config = config

        if not isinstance(tokenizer, bertgang_tokenizer.Tokenizer):
            raise TypeError(
                'parameter `tokenizer` must be type '
                '`bertgang.bertgang_tokenizer.Tokenizer`.'
            )

        self.tokenizer = tokenizer
        self.path = path

        all_dirs = bertgang_util.list_all_files(
            path,
            ignore_files=ignore_files
        )

        self.idx_dir_mapping = {}

        # Map each pre-train data by an unique id.
        # Each pre-train data is a directory, we will check if there is a
        # missing files.
        for idx, dir_name in enumerate(all_dirs):
            dir_path = f'{path}/{dir_name}'

            if is_checking_files:
                dir_files = bertgang_util.list_all_files(dir_path)

                # Check original sequence file.
                if 'original.pickle' not in dir_files:
                    raise ValueError(
                        f'Missing file `original.pickle` in {dir_path}.'
                    )

                # Check teacher tensor file.
                # Only need to check teachers in current configuration.
                for teacher in config.teachers:
                    if f'{teacher}.pickle' not in dir_files:
                        raise ValueError(
                            f'Missing file `{teacher}.pickle` in {dir_path}.'
                        )

            # Give a unique id to each pre-train data.
            self.idx_dir_mapping[idx] = dir_path

    def __len__(self) -> int:
        """Return pre-train data size."""
        return len(self.idx_dir_mapping)

    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, Union[str, torch.FloatTensor]]:
        """Return input tensors and knowledge distillation target tensors.

        Args:
            idx (int):
                Id of pre-train data.

        Returns:
            dict:
                See
                https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
                for more information.
                Must containing following keys:

                attention_mask:
                    A `torch.FloatTensor` with shape (B, S).
                input_ids:
                    A `torch.LongTensor` with shape (B, S).
                token_type_ids:
                    A `torch.LongTensor` with shape (B, S).
                teacher_1_name_output_embeds:
                    A `torch.FloatTensor` with shape (B, S).
                teacher_n_name_output_embeds:
                    Same as `teacher_1_name_output_embeds`.
        """

        target_dir = self.idx_dir_mapping[idx]

        with open(f'{target_dir}/original.pickle', 'rb') as input_file:
            input_obj = pickle.load(input_file)

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

        # TODO: Not sure if `obj['attention_mask']` is type
        # `torch.FloatTensor`, `obj['input_ids']` is type `torch.LongTensor`,
        # `obj['input_ids']` is type `torch.LongTensor`.

        for teacher in self.config.teachers:
            with open(f'{target_dir}/{teacher}.pickle', 'rb') as output_file:
                obj[f'{teacher}_output_embeds'] = torch.FloatTensor(
                    pickle.load(output_file)
                )

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
