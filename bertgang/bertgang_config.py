"""Configuration classes for pre-train and fine-tune experiments.

Use `bertgang.bertgang_config.PreTrainConfig` to setup pre-train experiments.
Use `bertgang.bertgang_config.FineTuneConfig` to setup fine-tune experiments.

Attributes:

    VALID_TEACHERS (dict):
        Key and value pairs, where key is the teacher name, and the value must
        contain the following informations:

        tokenizer (type(transformers.PreTrainedTokenizer)):
            A `transformers.PreTrainedTokenizer` subclass.
        version (str):
            Model version used by the following methods:

            - `transformers.PretrainedConfig.from_pretrained`
            - `transformers.PretrainedModel.from_pretrained`
            - `transformers.PreTrainedTokenizer.from_pretrained`

            Valid pre-trained version list can be found at the following link:
            https://huggingface.co/transformers/pretrained_models.html
        sp_indicator (str):
            sentencepiece indicator of the tokenizer under a specified version.

TODO:
    - Add `bertgang.bertgang_config.FineTuneConfig` class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from typing import List
from typing import Type

import transformers


VALID_TEACHERS = {
    'bert': {
        'tokenizer': transformers.BertTokenizer,
        'version': 'bert-base-cased',
        'sp_indicator': r'##',
    },
    'roberta': {
        'tokenizer': transformers.RobertaTokenizer,
        'version': 'roberta-base',
        'sp_indicator': r'Ġ',
    },
    'albert': {
        'tokenizer': transformers.AlbertTokenizer,
        'version': 'albert-base-v2',
        'sp_indicator': r'▁',
    },
}


class PreTrainConfig(transformers.AlbertConfig):
    """Configuration class for pre-train experiments.

    `bertgang.bertgang_config.PreTrainConfig` is the parameter of the
    following classes' contructor:

    - `bertgang.bertgang_model.Model`
    - `bertgang.bertgang_data.PreTrainData`

    `bertgang.bertgang_config.PreTrainConfig` is the parameter of the following
    methods:

    - `bertgang.bertgang_tokenizer.Tokenizer.learn_from_teachers`

    `bertgang.bertgang_config.PreTrainConfig` must be compatible with both of
    the following classes so the super class can be changed to meet the needs:

    - `bertgang.bertgang_model.Model`
    - `bertgang.bertgang_tokenizer.Tokenizer`

    The following list contains tuples of compatible config, model and
    tokenizer:

    - (`transformers.BertConfig`,
       `transformers.BertModel`,
       `transformers.BertTokenizer`)
    - (`transformers.RobertaConfig`,
       `transformers.RobertaModel`,
       `transformers.RobertTokenizer`)
    - (`transformers.AlbertConfig`,
       `transformers.AlbertModel`,
       `transformers.AlbertTokenizer`)

    `bertgang.bertgang_config.PreTrainConfig` is currently a subclass of
    `transformers.AlbertConfig`.
    Parameters of `transformers.AlbertConfig` can be seen from
    https://huggingface.co/transformers/model_doc/albert.html#albertconfig
    , and will not be listed here.

    Args:
        batch_size (int):
            Batch size for pre-train experiments.
        do_lower_case (bool):
            Whether to convert input tokens to lower case.
        path (str):
            Directory path to save the following files:

            - configuration
                - `config.json`
            - tokenizer
                - `added_tokens.json`,
                - `speical_tokens_map.json`
                - `spiece.model`
                - `tokenizer_config.json`
            - model
                - checkpoints
        teachers (list[str]):
            Teachers to perform knowledge distillation.
            There must have at least 1 teacher, and teachers must be in the
            keys of module variable `VALID_TEACHERS`.

    Attributes:
        batch_size (int):
            Same as parameter `batch_size`.
            Mainly used by the method
            `bertgang.bertgang_data.PreTrainData.get_data_loader`.
        do_lower_case (bool):
            Same as parameter `do_lower_case`.
            Mainly used by the method
            `bertgang.bertgang_tokenizer.Tokenizer.learn_from_teachers`.
        path (str):
            Same as parameter `path`.
            Mainly used as a parameter of the following methods:

            - `bertgang.bertgang_config.PreTrainConfig.save_pretrained`
            - `bertgang.bertgang_config.PreTrainConfig.from_pretrained`
            - `bertgang.bertgang_tokenizer.Tokenizer.save_pretrained`
            - `bertgang.bertgang_tokenizer.Tokenizer.from_pretrained`
            - `bertgang.bertgang_model.Model.save_pretrained`
            - `bertgang.bertgang_model.Model.from_pretrained`
        teachers (list[str]):
            Same as parameter `teachers`.
            Mainly used by the following methods:

            - `bertgang.bertgang_tokenizer.Tokenizer.learn_from_teachers`
            - `bertgang.bertgang_data.PreTrainData.__getitem__`

    Raises:
        TypeError:
            If `batch_size` is not type `int`,
            or `do_lower_case` is not type `bool`,
            or `path` is not type `str`,
            or `teacher` is not type `list`.
        ValueError:
            If `batch_size` is not a positive integer,
            or `teachers` is an empty list,
            or one of the teacher in `teachers` is not in `VALID_TEACHERS`.
        FileNotFoundError:
            If `path` does not exist.
        OSError:
            If `path` is not a directory.
    """

    def __init__(
            self,
            batch_size: int = None,
            do_lower_case: bool = None,
            path: str = None,
            teachers: List[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # check parameters
        if not isinstance(batch_size, int):
            raise TypeError('parameter `batch_size` must be type `int`.')

        if batch_size <= 0:
            raise ValueError(
                'parameter `batch_size` must be a positive integer.')

        self.batch_size = batch_size

        if not isinstance(do_lower_case, bool):
            raise TypeError('parameter `do_lower_case` must be type `bool`.')

        self.do_lower_case = do_lower_case

        if not isinstance(path, str):
            raise TypeError('parameter `path` must be type `str`.')

        if not os.path.exists(path):
            raise FileNotFoundError(f'directory `{path}` does not exist.')

        if not os.path.isdir(path):
            raise OSError(f'{path} is not a directory.')

        self.path = path

        if not isinstance(teachers, list):
            raise TypeError('parameter `teachers` must be type `list`.')

        if len(teachers) == 0:
            raise ValueError(
                'parameter `teachers` must have at least one teacher.'
            )

        for teacher in teachers:
            if teacher not in VALID_TEACHERS:
                raise ValueError(f'{teacher} is not a valid teacher.')

        self.teachers = teachers

    def is_teacher(
            self,
            teacher: str
    ) -> bool:
        """Check if `teacher` is in the current configuration teacher list.

        Args:
            teacher (str):
                Query teacher name.

        Returns:
            bool:
                True if `teacher` is in the current configuration teacher list,
                False otherwise.
        """

        return teacher in self.teachers

    def get_teacher_tokenizer(
            self,
            teacher: str
    ) -> Type[transformers.PreTrainedTokenizer]:
        """Get tokenizer class of `teacher`.

        Args:
            teacher (str):
                Query teacher name.

        Raises:
            ValueError:
                If `teacher` is not in the current configuration teacher list.

        Returns:
            type(transformers.PreTrainedTokenizer):
                `transformers.PreTrainedTokenizer` subclass matching with
                `teacher`.
                See module variable `VALID_TEACHER` for returned tokenizer
                classes.
        """

        if not self.is_teacher(teacher):
            raise ValueError(f'{teacher} is not a teacher.')
        return VALID_TEACHERS[teacher]['tokenizer']

    def get_teacher_version(
            self,
            teacher: str
    ) -> str:
        """Get pre-trained version of `teacher`.

        Args:
            teacher (str):
                Query teacher name.

        Raises:
            ValueError:
                If `teacher` is not in the current configuration teacher list.

        Returns:
            str:
                Model verion matching with `teacher`.
                See module variable `VALID_TEACHER` for returned model version.
        """

        if not self.is_teacher(teacher):
            raise ValueError(f'{teacher} is not a teacher.')
        return VALID_TEACHERS[teacher]['version']

    def get_teacher_tokenizer_instance(
            self,
            teacher: str
    ) -> transformers.PreTrainedTokenizer:
        """Get tokenizer instance of `teacher`.

        Args:
            teacher (str):
                Query teacher name.

        Raises:
            ValueError:
                If `teacher` is not in the current configuration teacher list.

        Returns:
            transformers.PreTrainedTokenizer:
                `transformers.PreTrainedTokenizer` subclass instance matching
                with `teacher`.
                See module variable `VALID_TEACHER` for returned tokenizer
                classes.
        """

        tokenizer = self.get_teacher_tokenizer(teacher)
        version = self.get_teacher_version(teacher)
        return tokenizer.from_pretrained(version)

    def get_teacher_sp_indicator(
            self,
            teacher: str
    ) -> str:
        """Get pre-trained tokenizer sentencepiece indicator of `teacher`.

        Args:
            teacher (str):
                Query teacher name.

        Raises:
            ValueError:
                If `teacher` is not in the current configuration teacher list.

        Returns:
            str:
                sentencepiece indicator matching with `teacher`.
                See module variable `VALID_TEACHER` for returned sentencepiece
                indicator.
        """

        if not self.is_teacher(teacher):
            raise ValueError(f'{teacher} is not a teacher.')
        return VALID_TEACHERS[teacher]['sp_indicator']
