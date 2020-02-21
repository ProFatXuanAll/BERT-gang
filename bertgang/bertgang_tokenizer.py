"""Tokenizer class for both pre-train and fine-tune experiments.

TODO:
    - Train a sentencepiece model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

import transformers

from . import bertgang_config


class Tokenizer(transformers.AlbertTokenizer):
    """Tokenizer class for both pre-train and fine-tune experiments.

    `bertgang.bertgang_tokenizer.Tokenizer` is the parameter of the following
    classes' contructor:

    - `bertgang.bertgang_data.PreTrainData`
    - `bertgang.bertgang_data.FineTuneData`

    `bertgang.bertgang_tokenizer.Tokenizer` must be compatible with both of the
    following classes so the super class can be changed to meet the needs:

    - `bertgang.bertgang_model.Model`
    - `bertgang.bertgang_config.PreTrainConfig`

    See `bertgang.bertgang_config.PreTrainConfig` for full compatible list.

    `bertgang.bertgang_tokenizer.Tokenizer` is currently a subclass of
    `transformers.AlbertTokenizer`.
    Parameters of `transformers.AlbertTokenizer` can be seen from
    https://huggingface.co/transformers/model_doc/albert.html#alberttokenizer
    , and will not be listed here.

    TODO:
        - Train a sentencepiece model
        - Fix method learn_from_teachers
    """

    def learn_from_teachers(
            self,
            config: bertgang_config.PreTrainConfig,
            **kwargs
    ) -> None:
        """Learn sentencepiece from all teachers.

        Args:
            config (bertgang_config.PreTrainConfig):
                Source of teachers list.

        Raises:
            TypeError:
                If `config` is not type `bertgang_config.PreTrainConfig`.

        TODO: This method does not work as expected.
        """

        # check parameters
        if not isinstance(config, bertgang_config.PreTrainConfig):
            raise TypeError(
                'parameter `config` must be type '
                '`bertgang.bertgang_config.PreTrainConfig`.'
            )

        albert_sp_indicator = (bertgang_config
                               .VALID_TEACHERS['albert']['sp_indicator'])

        for teacher in config.teachers:
            teacher_tokenizer = config.get_teacher_tokenizer_instance(teacher)
            teacher_sp_indicator = config.get_teacher_sp_indicator(teacher)

            for idx in range(teacher_tokenizer.vocab_size):
                if idx in teacher_tokenizer.all_special_ids:
                    continue

                token = teacher_tokenizer.convert_ids_to_tokens([idx])[0]
                token = re.sub(teacher_sp_indicator, '', token)
                if config.do_lower_case:
                    token = token.lower()
                token = albert_sp_indicator + token

                # TODO: this does not change the actual sentencepiece
                # model, so we must train a sentencepiece model.
                self.add_tokens([token])
