from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
from typing import Optional

import transformers


class PreTrainConfig(transformers.AlbertConfig):

    valid_teachers = {
        'bert': {
            'tokenizer': transformers.BertTokenizer,
            'version': 'bert-base-cased',
            'sp_indicator': r'##',
        },
        'roberta': {
            'tokenizer': transformers.RobertaTokenizer,
            'version': 'roberta-base',
            'sp_indicator': r'</w>',
        },
        'albert': {
            'tokenizer': transformers.AlbertTokenizer,
            'version': 'albert-base-v2',
            'sp_indicator': r'▁',
        },
    }

    def __init__(
        self,
        attention_probs_dropout_prob: float = 0.0,
        batch_size: int = 9,
        classifier_dropout_prob: float = 0.1,
        do_lower_case: bool = True,
        embedding_size: int = 128,
        hidden_act: str = 'gelu_new',
        hidden_dropout_prob: float = 0.0,
        hidden_size: int = 768,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        inner_group_num: int = 1,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 512,
        num_attention_heads: int = 12,
        num_hidden_groups: int = 1,
        num_hidden_layers: int = 6,
        teachers: List[str] = ['bert', 'roberta', 'albert', ],
        type_vocab_size: int = 2,
        vocab_size: int = 30000,
        **kwargs
    ):
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            embedding_size=embedding_size,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            inner_group_num=inner_group_num,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            num_hidden_groups=num_hidden_groups,
            num_attention_heads=num_attention_heads,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            **kwargs
        )
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

        if not isinstance(teachers, list):
            raise TypeError('parameter `teachers` must be type `list`.')

        if len(teachers) == 0:
            raise ValueError(
                'parameter `teachers` must have at least one teacher.')

        for teacher in teachers:
            if not self.is_valid_teacher(teacher):
                raise ValueError(f'{teacher} is not a valid teacher.')

        self.teachers = teachers

    def is_valid_teacher(self, teacher):
        return teacher in self.__class__.valid_teachers

    def get_teacher_tokenizer(self, teacher):
        if not self.is_valid_teacher(teacher):
            raise ValueError(f'{teacher} is not a valid teacher.')
        return self.__class__.valid_teachers[teacher]['tokenizer']

    def get_teacher_version(self, teacher):
        if not self.is_valid_teacher(teacher):
            raise ValueError(f'{teacher} is not a valid teacher.')
        return self.__class__.valid_teachers[teacher]['version']

    def get_teacher_tokenizer_instance(self, teacher):
        tokenizer = self.get_teacher_tokenizer(teacher)
        version = self.get_teacher_version(teacher)
        return tokenizer.from_pretrained(version)

    def get_teacher_sp_indicator(self, teacher):
        if not self.is_valid_teacher(teacher):
            raise ValueError(f'{teacher} is not a valid teacher.')
        return self.__class__.valid_teachers[teacher]['sp_indicator']