from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

from typing import List
from typing import Optional

import transformers

from . import bertgang_config


class Tokenizer(transformers.AlbertTokenizer):
    def learn_from_teachers(
        self,
        config: bertgang_config.PreTrainConfig,
        **kwargs
    ) -> None:
        if not isinstance(config, bertgang_config.PreTrainConfig):
            raise TypeError(
                'parameter `config` must be type `bertgang_config.PreTrainConfig`.')

        albert_sp_indicator = config.get_teacher_sp_indicator('albert')
        for teacher in config.teachers:
            teacher_tokenizer = config.get_teacher_tokenizer_instance(teacher)
            teacher_sp_indicator = config.get_teacher_sp_indicator(teacher)
            for idx in range(teacher_tokenizer.vocab_size):
                if idx in teacher_tokenizer.all_special_ids:
                    continue
                else:
                    token = teacher_tokenizer.convert_ids_to_tokens([idx])[0]
                    token = re.sub(teacher_sp_indicator, '', token)
                    if config.do_lower_case:
                        token = token.lower()
                    token = albert_sp_indicator + token
                    # TODO: this does not change the actual sentencepiece model
                    self.add_tokens([token])
