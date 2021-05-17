r"""Helper functions for loading model.

Usage:
    import fine_tune

    teacher_model = fine_tune.util.load_teacher_model(...)
    teacher_model = fine_tune.util.load_teacher_model_by_config(...)

    student_model = fine_tune.util.load_student_model(...)
    student_model = fine_tune.util.load_student_model_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import transformers

# my own modules

import fine_tune.config
import fine_tune.model


def load_student_model(
        d_emb: int,
        d_ff: int,
        d_model: int,
        device: torch.device,
        dropout: float,
        max_seq_len: int,
        model: str,
        num_attention_heads: int,
        num_class: int,
        num_hidden_layers: int,
        vocab_size: int,
        type_vocab_size: int,
        init_from_pre_trained: bool
) -> fine_tune.model.StudentModel:
    r"""Load student model.

    Args:
        d_emb:
            Embedding dimension.
        d_ff:
            Transformer layers feed forward dimension.
        d_model:
            Transformer layers hidden dimension.
        device:
            Student model running device.
        dropout:
            Dropout probability.
        max_seq_len:
            Maximum input sequence length for fine-tune model.
        model:
            Student model name.
        num_attention_heads:
            Number of attention heads in Transformer layers.
        num_class:
            Number of classes to classify.
        num_hidden_layers:
            Number of Transformer layers.
        type_vocab_size:
            BERT-like models token type embedding range.
        vocab_size:
            Vocabulary dimension.
        init_from_pre_trained:
            Use pre-trained model wieght to init student model.

    Raises:
        ValueError:
            If `model` does not supported.

    Returns:
        `fine_tune.model.StudentAlbert`:
            If `config.student` is 'albert'.
        `fine_tune.model.StudentBert`:
            If `config.student` is 'bert'.
    """
    if model == 'albert':
        return fine_tune.model.StudentAlbert(
            d_emb=d_emb,
            d_ff=d_ff,
            d_model=d_model,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_attention_heads=num_attention_heads,
            num_class=num_class,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size
        ).to(device)

    if model == 'bert':
        return fine_tune.model.StudentBert(
            d_ff=d_ff,
            d_model=d_model,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_attention_heads=num_attention_heads,
            num_class=num_class,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            init_from_pre_trained=init_from_pre_trained
        ).to(device)

    raise ValueError(
        f'`model` {model} is not supported.\nSupported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--model {option}',
            [
                'bert',
                'albert'
            ]
        )))
    )


def load_student_model_by_config(
        config: fine_tune.config.StudentConfig,
        tokenizer: transformers.PreTrainedTokenizer,
        init_from_pre_trained: bool
) -> fine_tune.model.StudentModel:
    r"""Load student model.

    Args:
        config:
            `fine_tune.config.StudentConfig` which contains attributes `d_emb`,
            `d_ff`, `d_model`, `device`, `dropout`, `max_seq_len`, `model`,
            `num_attention_heads`, `num_class`, `num_hidden_layers` and
            `type_vocab_size`.
        tokenizer:
            Tokenizer object which contains attribute `vocab_size`.
        init_from_pre_trained:
            Use pre-trained model wieght to init student model.

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
        model=config.model,
        num_attention_heads=config.num_attention_heads,
        num_class=config.num_class,
        num_hidden_layers=config.num_hidden_layers,
        type_vocab_size=config.type_vocab_size,
        vocab_size=tokenizer.vocab_size,
        init_from_pre_trained=init_from_pre_trained
    )


def load_teacher_model(
        device: torch.device,
        dropout: float,
        model: str,
        num_class: int,
        ptrain_ver: str
) -> fine_tune.model.TeacherModel:
    r"""Load teacher model.

    Args:
        device:
            Teacher model running device.
        dropout:
            Dropout probability.
        model:
            Teacher model name.
        num_class:
            Number of classes to classify.
        ptrain_ver:
            Pretrained model version provided by `transformers` package.

    Raises:
        ValueError:
            If `model` does not supported.

    Returns:
        `fine_tune.model.TeacherAlbert`:
            If `config.model` is 'albert'.
        `fine_tune.model.TeacherBert`:
            If `config.model` is 'bert'.
    """
    if model == 'albert':
        return fine_tune.model.TeacherAlbert(
            dropout=dropout,
            num_class=num_class,
            ptrain_ver=ptrain_ver
        ).to(device)

    if model == 'bert':
        return fine_tune.model.TeacherBert(
            dropout=dropout,
            num_class=num_class,
            ptrain_ver=ptrain_ver
        ).to(device)

    raise ValueError(
        f'`model` {model} is not supported.\n' +
        'Supported options:' +
        ''.join(list(map(
            lambda option: f'\n\t--model {option}',
            [
                'bert',
                'albert'
            ]
        )))
    )


def load_teacher_model_by_config(
        config: fine_tune.config.TeacherConfig
) -> fine_tune.model.TeacherModel:
    r"""Load teacher model.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which contains attributes
            `device`, `dropout`, `model`, `num_class` and `ptrain_ver`.

    Returns:
        Same as `fine_tune.util.load_teacher_model`.
    """
    return load_teacher_model(
        device=config.device,
        dropout=config.dropout,
        model=config.model,
        num_class=config.num_class,
        ptrain_ver=config.ptrain_ver
    )
