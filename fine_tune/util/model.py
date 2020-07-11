r"""Helper function for loading model.

Usage:
    model = fine_tune.util.load_teacher_model(...)
    model = fine_tune.util.load_teacher_model_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# 3rd party modules

import torch
import transformers

# my own modules

import fine_tune.config
import fine_tune.model


def load_teacher_model(
        device: torch.device,
        dropout: float,
        num_class: int,
        num_gpu: int,
        pretrained_version: str,
        teacher: str,
) -> Union[
    fine_tune.model.TeacherAlbert,
    fine_tune.model.TeacherBert,
]:
    r"""Load teacher model.

    Args:
        device:
            CUDA device to run model.
        dropout:
            Dropout probability.
        num_class:
            Number of classes to classify.
        num_gpu:
            Number of GPUs to train.
        pretrained_version:
            Pretrained model provided by hugginface.
        teacher:
            Teacher model's name.

    Returns:
        `fine_tune.model.TeacherAlbert`:
            If `config.teacher` is 'albert'.
        `fine_tune.model.TeacherBert`:
            If `config.teacher` is 'bert'.
    """

    if teacher == 'albert':
        model = fine_tune.model.TeacherAlbert(
            dropout=dropout,
            num_class=num_class,
            pretrained_version=pretrained_version
        )
    if teacher == 'bert':
        model = fine_tune.model.TeacherBert(
            dropout=dropout,
            num_class=num_class,
            pretrained_version=pretrained_version
        )

    if num_gpu >= 1:
        model = model.to(device)

    return model


def load_teacher_model_by_config(
        config: fine_tune.config.TeacherConfig
) -> Union[
    fine_tune.model.TeacherAlbert,
    fine_tune.model.TeacherBert,
]:
    r"""Load teacher model.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `device`, `dropout`, `num_class`, `num_gpu`, `pretrained_version`
            and `teacher`.

    Returns:
        Same as `fine_tune.util.load_teacher_model`.
    """
    return load_teacher_model(
        device=config.device,
        dropout=config.dropout,
        num_class=config.num_class,
        num_gpu=config.num_gpu,
        pretrained_version=config.pretrained_version,
        teacher=config.teacher
    )


def load_student_model(
        d_emb: int,
        d_ff: int,
        d_model: int,
        device: torch.device,
        dropout: float,
        max_seq_len: int,
        num_attention_heads: int,
        num_class: int,
        num_gpu: int,
        num_hidden_layers: int,
        student: str,
        tokenizer: Union[
            transformers.AlbertTokenizer,
            transformers.BertTokenizer,
        ],
        type_vocab_size: int
) -> Union[
    fine_tune.model.StudentAlbert,
    fine_tune.model.StudentBert,
]:
    r"""Load student model.

    Args:
        d_emb:
            Embedding dimension.
        d_ff:
            Transformer layers feed forward dimension.
        d_model:
            Transformer layers hidden dimension.
        device:
            CUDA device to run model.
        dropout:
            Dropout probability.
        max_seq_len:
            Max sequence length for fine-tune.
        num_attention_heads:
            Number of attention heads in Transformer layers.
        num_class:
            Number of classes to classify.
        num_gpu:
            Number of GPUs to train.
        num_hidden_layers:
            Number of Transformer layers.
        student:
            Student model's name.
        tokenizer:
            Tokenizer object which contains attribute `vocab_size`.
        type_vocab_size:
            BERT-like models token type embedding range.

    Returns:
        `fine_tune.model.StudentAlbert`:
            If `config.student` is 'albert'.
        `fine_tune.model.StudentBert`:
            If `config.student` is 'bert'.
    """

    if student == 'albert':
        model = fine_tune.model.StudentAlbert(
            d_emb=d_emb,
            d_ff=d_ff,
            d_model=d_model,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_attention_heads=num_attention_heads,
            num_class=num_class,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            vocab_size=tokenizer.vocab_size
        )
    if student == 'bert':
        model = fine_tune.model.StudentBert(
            d_ff=d_ff,
            d_model=d_model,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_attention_heads=num_attention_heads,
            num_class=num_class,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            vocab_size=tokenizer.vocab_size
        )

    if num_gpu >= 1:
        model = model.to(device)

    return model


def load_student_model_by_config(
        config: fine_tune.config.StudentConfig,
        tokenizer: Union[
            transformers.AlbertTokenizer,
            transformers.BertTokenizer,
        ]
) -> Union[
    fine_tune.model.StudentAlbert,
    fine_tune.model.StudentBert,
]:
    r"""Load student model.

    Args:
        config:
            `fine_tune.config.StudentConfig` which contains attributes `d_emb`,
            `d_ff`, `d_model`, `device`, `dropout`, `max_seq_len`,
            `num_attention_heads`, `num_class`, `num_gpu`, `num_hidden_layers`,
            `student` and `type_vocab_size`.
        tokenizer:
            Tokenizer object which contains attribute `vocab_size`.

    Returns:
        Same as `fine_tune.util.load_student_model`.
    """
    return load_student_model(
        d_emb=config.d_emb,
        d_ff=config.d_ff,
        d_model=config.d_model,
        device=config.device,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        num_attention_heads=config.num_attention_heads,
        num_class=config.num_class,
        num_gpu=config.num_gpu,
        num_hidden_layers=config.num_hidden_layers,
        student=config.student,
        tokenizer=tokenizer,
        type_vocab_size=config.type_vocab_size
    )
