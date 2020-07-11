r"""Configuration for fine-tune experiments.

Usage:
    teacher_config = TeacherConfig(...params)
    teacher_config.save()
    teacher_config = TeacherConfig.load(...params)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

from typing import Generator
from typing import Tuple
from typing import Union

# 3rd party modules

import torch

# my own modules

import fine_tune.path


class TeacherConfig:
    r"""Configuration for fine-tune teacher model.

    All teacher model will be optimized with AdamW.
    Optimization will be paired with linear warmup scheduler.

    Attributes:
        accumulation_step:
            Gradient accumulation step.
            Used when GPU memory cannot fit in whole batch.
            Minimum is `1`.
            Must be smaller than or equal to `batch_size`.
        allow_tasks:
            Currently supported tasks.
            Must be updated when adding more supported tasks.
        allow_teachers:
            Currently supported teacher models.
            Must be updated when adding more supported teacher models.
        batch_size:
            Training batch size.
            Minimum is `1`.
        beta1:
            Optimizer AdamW's parameter `betas` first value.
            Must be ranging from `0` to `1` (inclusive).
        beta2:
            Optimizer AdamW's parameter `betas` second value.
            Must be ranging from `0` to `1` (inclusive).
        checkpoint_step:
            Checkpoint interval based on number of mini-batch.
            Must be bigger than or equal to `1`.
        dataset:
            Dataset name of particular fine-tune task.
            For example, task `MNLI` have dataset 'train',
            'dev_matched' and 'dev_mismatched'.
        dropout:
            Dropout probability.
            Must be ranging from `0` to `1` (inclusive).
        epoch:
            Number of training epochs.
            Must be bigger than or equal to `1`.
        eps:
            Optimizer AdamW's parameter `eps`.
            Must be bigger than `0`.
        experiment:
            Name of current experiment.
        learning_rate:
            Optimizer AdamW's parameter `lr`.
            Must be bigger than `0`.
        max_norm:
            Max norm of gradient.
            Used when cliping gradient norm.
            Must be bigger than `0`.
        max_seq_len:
            Max sequence length for fine-tune.
            Must be bigger than `0`.
        num_class:
            Number of classes to classify.
            Must be bigger than `1`.
        num_gpu:
            Number of GPUs to train.
            Must be bigger than or equal to `0`.
        pretrained_version:
            Pretrained model provided by hugginface.
            This value will be check again in `fine_tune/model.py`
        seed:
            Control random seed.
            Must be bigger than `0`.
        task:
            Name of fine-tune task.
            `task` must be in `TeacherConfig.allow_tasks`.
        teacher:
            Teacher model's name.
            `teacher` must be in `TeacherConfig.allow_teachers`.
        warmup_step
            Linear scheduler warmup step.
            Must be bigger than or equal to `0`.
        weight_decay:
            Optimizer AdamW's parameter `weight_decay`.
            Must be bigger than or equal to `0`.

    Raises:
        ValueError:
            If constrains on any attributes failed.
            See attributes section for details.
        OSError:
            If `num_gpu > 0` and no CUDA device are available.
    """

    allow_tasks = (
        'mnli',
    )

    allow_teachers = (
        'albert',
        'bert',
    )

    def __init__(
            self,
            accumulation_step: int = 1,
            batch_size: int = 32,
            beta1: float = 0.9,
            beta2: float = 0.999,
            checkpoint_step: int = 500,
            dataset: str = '',
            dropout: float = 0.1,
            epoch: int = 3,
            eps: float = 1e-8,
            experiment: str = '',
            learning_rate: float = 3e-5,
            max_norm: float = 1.0,
            max_seq_len: int = 512,
            num_class: int = 2,
            num_gpu: int = 0,
            pretrained_version: str = '',
            seed: int = 42,
            task: str = '',
            teacher: str = '',
            warmup_step: int = 10000,
            weight_decay: float = 0.01
    ):

        if accumulation_step < 1:
            raise ValueError(
                '`accumulation_step` must be bigger than or equal to `1`'
            )

        if batch_size < 1:
            raise ValueError(
                '`batch_size` must be bigger than or equal to `1`.'
            )

        if accumulation_step > batch_size:
            raise ValueError(
                '`accumulation_step` must be smaller than or equal to `batch_size`.'
            )

        if not 0 <= beta1 <= 1:
            raise ValueError(
                '`beta1` must be ranging from `0` to `1` (inclusive).'
            )

        if not 0 <= beta2 <= 1:
            raise ValueError(
                '`beta2` must be ranging from `0` to `1` (inclusive).'
            )

        if checkpoint_step < 1:
            raise ValueError(
                '`checkpoint_step` must be bigger than or equal to `1`.'
            )

        if not 0 <= dropout <= 1:
            raise ValueError(
                '`dropout` must be ranging from `0` to `1` (inclusive).'
            )

        if epoch < 1:
            raise ValueError(
                '`epoch` must be bigger than or equal to `1`.'
            )

        if eps <= 0:
            raise ValueError(
                '`eps` must be bigger than `0`.'
            )

        if learning_rate <= 0:
            raise ValueError(
                '`learning_rate` must be bigger than `0`.'
            )

        if max_norm <= 0:
            raise ValueError(
                '`max_norm` must be bigger than `0`.'
            )

        if max_seq_len <= 0:
            raise ValueError(
                '`max_seq_len` must be bigger than `0`.'
            )

        if num_class <= 1:
            raise ValueError(
                '`num_class` must be bigger than `1`.'
            )

        if num_gpu < 0:
            raise ValueError(
                '`num_gpu` must be bigger than or equal to `0`.'
            )

        if num_gpu > 0 and not torch.cuda.is_available():
            raise OSError(
                'CUDA device not found, set `num_gpu` to `0`.'
            )

        if seed <= 0:
            raise ValueError(
                '`seed` must be bigger than `0`.'
            )

        if task not in TeacherConfig.allow_tasks:
            raise ValueError(
                '`task` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        self.__class__.allow_tasks
                    )
                )
            )

        if teacher not in TeacherConfig.allow_teachers:
            raise ValueError(
                '`teacher` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        TeacherConfig.allow_teachers
                    )
                )
            )

        if warmup_step < 0:
            raise ValueError(
                '`warmup_step` must be bigger than or equal to `0`.'
            )

        if weight_decay < 0:
            raise ValueError(
                '`weight_decay` must be bigger than or equal to `0`.'
            )

        self.accumulation_step = accumulation_step
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.checkpoint_step = checkpoint_step
        self.dataset = dataset
        self.dropout = dropout
        self.epoch = epoch
        self.eps = eps
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.max_norm = max_norm
        self.max_seq_len = max_seq_len
        self.num_class = num_class
        self.num_gpu = num_gpu
        self.pretrained_version = pretrained_version
        self.seed = seed
        self.task = task
        self.teacher = teacher
        self.warmup_step = warmup_step
        self.weight_decay = weight_decay

    def __iter__(self) -> Generator[
            Tuple[str, Union[float, int, str]], None, None
    ]:
        yield 'accumulation_step', self.accumulation_step
        yield 'batch_size', self.batch_size
        yield 'beta1', self.beta1
        yield 'beta2', self.beta2
        yield 'checkpoint_step', self.checkpoint_step
        yield 'dataset', self.dataset
        yield 'dropout', self.dropout
        yield 'epoch', self.epoch
        yield 'eps', self.eps
        yield 'experiment', self.experiment
        yield 'learning_rate', self.learning_rate
        yield 'max_norm', self.max_norm
        yield 'max_seq_len', self.max_seq_len
        yield 'num_class', self.num_class
        yield 'num_gpu', self.num_gpu
        yield 'pretrained_version', self.pretrained_version
        yield 'seed', self.seed
        yield 'task', self.task
        yield 'teacher', self.teacher
        yield 'warmup_step', self.warmup_step
        yield 'weight_decay', self.weight_decay

    def __str__(self) -> str:
        col_width = max([
            max(len(k), len(str(v)))
            for k, v in self
        ])
        table_width = 2 * (col_width + 2) + 1
        sep = '\n+' + '-' * table_width + '+'
        row = '\n| {:<{col_width}} | {:<{col_width}} |'
        table = (
            sep +
            row.format('configuration', 'value', col_width=col_width) +
            sep +
            ''.join([
                row.format(k, v, col_width=col_width)
                for k, v in self
            ]) +
            sep
        )

        return table

    @property
    def betas(self) -> Tuple[float, float]:
        r"""Optimizer AdamW's parameter `betas`.

        Returns:
            A tuple contain two values, `self.beta1, self.beta2`.
        """
        return self.beta1, self.beta2

    @property
    def device(self) -> torch.device:
        r"""Get running model device.

        If `self.num_gpu == 0`, then run model on CPU.
        Else run model on CUDA device.

        Returns:
            Device create by `torch.device`.
        """
        if not self.num_gpu:
            return torch.device('cpu')
        return torch.device('cuda')

    def save(self) -> None:
        r"""Save configuration into json file."""
        file_path = TeacherConfig.file_path(
            experiment=self.experiment,
            task=self.task,
            teacher=self.teacher
        )

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as json_file:
            json.dump(
                dict(self),
                json_file,
                ensure_ascii=False
            )

    @classmethod
    def load(
            cls,
            experiment: str = '',
            task: str = '',
            teacher: str = ''
    ):
        r"""Load configuration from json file.

        Args:
        experiment:
            Name of experiment.
        task:
            Name of fine-tune task.
        teacher:
            Teacher model's name.
        """
        file_path = TeacherConfig.file_path(
            experiment=experiment,
            task=task,
            teacher=teacher
        )
        with open(file_path, 'r') as json_file:
            return cls(**json.load(json_file))

    @staticmethod
    def format_experiment_name(
            experiment: str = '',
            task: str = '',
            teacher: str = ''
    ) -> str:
        r"""Return formatted experiment name.

        Args:
            experiment:
                Name of experiment.
            task:
                Name of fine-tune task.
            teacher:
                Teacher model's name.

        Returns:
            A string of experiment name.
        """
        return f'{teacher}-{task}-{experiment}'

    @staticmethod
    def file_path(
            experiment: str = '',
            task: str = '',
            teacher: str = ''
    ) -> str:
        r"""Return formatted config file name.

        Args:
            experiment:
                Name of experiment.
            task:
                Name of fine-tune task.
            teacher:
                Teacher model's name.

        Returns:
            A string of config file path.
        """
        return '{}/{}/{}'.format(
            fine_tune.path.FINE_TUNE_EXPERIMENT,
            TeacherConfig.format_experiment_name(
                experiment=experiment,
                task=task,
                teacher=teacher
            ),
            fine_tune.path.CONFIG
        )


class StudentConfig:
    r"""Configuration for fine-tune distill student model.

    All student model will be optimized with AdamW.
    Optimization will be paired with linear warmup scheduler.

    Attributes:
        accumulation_step:
            Gradient accumulation step.
            Used when GPU memory cannot fit in whole batch.
            Minimum is `1`.
            Must be smaller than or equal to `batch_size`.
        allow_students:
            Currently supported student models.
            Must be updated when adding more supported student models.
        allow_tasks:
            Currently supported tasks.
            Must be updated when adding more supported tasks.
        batch_size:
            Training batch size.
            Minimum is `1`.
        beta1:
            Optimizer AdamW's parameter `betas` first value.
            Must be ranging from `0` to `1` (inclusive).
        beta2:
            Optimizer AdamW's parameter `betas` second value.
            Must be ranging from `0` to `1` (inclusive).
        checkpoint_step:
            Checkpoint interval based on number of mini-batch.
            Must be bigger than or equal to `1`.
        d_emb:
            Embedding dimension.
            Must be bigger than or equal to `1`.
        d_ff:
            Transformer layers feed forward dimension.
            Must be bigger than `0`.
        d_model:
            Transformer layers hidden dimension.
            Must be bigger than `0`.
        dataset:
            Name of the distillation dataset generated by teacher model.
            Raise `ValueError` if teacher fine-tune experiment does not exist.
        dropout:
            Dropout probability.
            Must be ranging from `0` to `1` (inclusive).
        epoch:
            Number of training epochs.
            Must be bigger than or equal to `1`.
        eps:
            Optimizer AdamW's parameter `eps`.
            Must be bigger than `0`.
        experiment:
            Name of current experiment.
        learning_rate:
            Optimizer AdamW's parameter `lr`.
            Must be bigger than `0`.
        max_norm:
            Max norm of gradient.
            Used when cliping gradient norm.
            Must be bigger than `0`.
        max_seq_len:
            Max sequence length for fine-tune.
            Must be bigger than `0`.
        num_attention_heads:
            Number of attention heads in Transformer layers.
            Must be bigger than `0`.
        num_class:
            Number of classes to classify.
            Must be bigger than `1`.
        num_gpu:
            Number of GPUs to train.
            Must be bigger than or equal to `0`.
        num_hidden_layers:
            Number of Transformer layers.
            Must be bigger than or equal to `1`.
        seed:
            Control random seed.
            Must be bigger than `0`.
        student:
            Student model's name.
            `student` must be in `StudentConfig.allow_students`.
        task:
            Name of fine-tune task.
            `task` must be in `StudentConfig.allow_tasks`.
        type_vocab_size:
            BERT-like models token type embedding range.
            Must be bigger than `0`.
        warmup_step
            Linear scheduler warmup step.
            Must be bigger than or equal to `0`.
        weight_decay:
            Optimizer AdamW's parameter `weight_decay`.
            Must be bigger than or equal to `0`.

    Raises:
        ValueError:
            If constrains on any attributes failed.
            See attributes section for details.
        OSError:
            If `num_gpu > 0` and no CUDA device are available.
    """

    allow_tasks = (
        'mnli',
    )

    allow_students = (
        'albert',
        'bert',
    )

    def __init__(
            self,
            accumulation_step: int = 1,
            batch_size: int = 32,
            beta1: float = 0.9,
            beta2: float = 0.999,
            checkpoint_step: int = 500,
            d_emb: int = 128,
            d_ff: int = 3072,
            d_model: int = 768,
            dataset: str = '',
            dropout: float = 0.1,
            epoch: int = 3,
            eps: float = 1e-8,
            experiment: str = '',
            learning_rate: float = 3e-5,
            max_norm: float = 1.0,
            max_seq_len: int = 512,
            num_attention_heads: int = 16,
            num_class: int = 2,
            num_gpu: int = 0,
            num_hidden_layers: int = 6,
            seed: int = 42,
            student: str = '',
            task: str = '',
            type_vocab_size: int = 2,
            warmup_step: int = 10000,
            weight_decay: float = 0.01
    ):

        if accumulation_step < 1:
            raise ValueError(
                '`accumulation_step` must be bigger than or equal to `1`'
            )

        if batch_size < 1:
            raise ValueError(
                '`batch_size` must be bigger than or equal to `1`.'
            )

        if accumulation_step > batch_size:
            raise ValueError(
                '`accumulation_step` must be smaller than or equal to `batch_size`.'
            )

        if not 0 <= beta1 <= 1:
            raise ValueError(
                '`beta1` must be ranging from `0` to `1` (inclusive).'
            )

        if not 0 <= beta2 <= 1:
            raise ValueError(
                '`beta2` must be ranging from `0` to `1` (inclusive).'
            )

        if checkpoint_step < 1:
            raise ValueError(
                '`checkpoint_step` must be bigger than or equal to `1`.'
            )

        if d_emb < 1:
            raise ValueError(
                '`d_emb` must be bigger than or equal to `1`.'
            )

        if d_ff <= 0:
            raise ValueError(
                '`d_ff` must be bigger than `0`.'
            )

        if d_model <= 0:
            raise ValueError(
                '`d_model` must be bigger than `0`.'
            )

        if not os.path.exists(
                f'{fine_tune.path.FINE_TUNE_EXPERIMENT}/{dataset}'
        ):
            raise ValueError(
                f'Fine-tune experiment {dataset} does not exist.'
            )

        if not 0 <= dropout <= 1:
            raise ValueError(
                '`dropout` must be ranging from `0` to `1` (inclusive).'
            )

        if epoch < 1:
            raise ValueError(
                '`epoch` must be bigger than or equal to `1`.'
            )

        if eps <= 0:
            raise ValueError(
                '`eps` must be bigger than `0`.'
            )

        if learning_rate <= 0:
            raise ValueError(
                '`learning_rate` must be bigger than `0`.'
            )

        if max_norm <= 0:
            raise ValueError(
                '`max_norm` must be bigger than `0`.'
            )

        if max_seq_len <= 0:
            raise ValueError(
                '`max_seq_len` must be bigger than `0`.'
            )

        if num_attention_heads <= 0:
            raise ValueError(
                '`num_attention_heads` must be bigger than `0`.'
            )

        if num_class <= 1:
            raise ValueError(
                '`num_class` must be bigger than `1`.'
            )

        if num_gpu < 0:
            raise ValueError(
                '`num_gpu` must be bigger than or equal to `0`.'
            )

        if num_gpu > 0 and not torch.cuda.is_available():
            raise OSError(
                'CUDA device not found, set `num_gpu` to `0`.'
            )

        if num_hidden_layers < 1:
            raise ValueError(
                '`num_hidden_layers` must be bigger than or equal to `1`.'
            )

        if seed <= 0:
            raise ValueError(
                '`seed` must be bigger than `0`.'
            )

        if student not in StudentConfig.allow_students:
            raise ValueError(
                '`student` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        StudentConfig.allow_students
                    )
                )
            )

        if task not in StudentConfig.allow_tasks:
            raise ValueError(
                '`task` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        self.__class__.allow_tasks
                    )
                )
            )

        if type_vocab_size <= 0:
            raise ValueError(
                '`type_vocab_size` must be bigger than `0`.'
            )

        if warmup_step < 0:
            raise ValueError(
                '`warmup_step` must be bigger than or equal to `0`.'
            )

        if weight_decay < 0:
            raise ValueError(
                '`weight_decay` must be bigger than or equal to `0`.'
            )

        self.accumulation_step = accumulation_step
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.checkpoint_step = checkpoint_step
        self.d_emb = d_emb
        self.d_ff = d_ff
        self.d_model = d_model
        self.dataset = dataset
        self.dropout = dropout
        self.epoch = epoch
        self.eps = eps
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.max_norm = max_norm
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads
        self.num_class = num_class
        self.num_gpu = num_gpu
        self.num_hidden_layers = num_hidden_layers
        self.seed = seed
        self.student = student
        self.task = task
        self.type_vocab_size = type_vocab_size
        self.warmup_step = warmup_step
        self.weight_decay = weight_decay

    def __iter__(self) -> Generator[
            Tuple[str, Union[float, int, str]], None, None
    ]:
        yield 'accumulation_step', self.accumulation_step
        yield 'batch_size', self.batch_size
        yield 'beta1', self.beta1
        yield 'beta2', self.beta2
        yield 'checkpoint_step', self.checkpoint_step
        yield 'd_emb', self.d_emb
        yield 'd_ff', self.d_ff
        yield 'd_model', self.d_model
        yield 'dataset', self.dataset
        yield 'dropout', self.dropout
        yield 'epoch', self.epoch
        yield 'eps', self.eps
        yield 'experiment', self.experiment
        yield 'learning_rate', self.learning_rate
        yield 'max_norm', self.max_norm
        yield 'max_seq_len', self.max_seq_len
        yield 'num_attention_heads', self.num_attention_heads
        yield 'num_class', self.num_class
        yield 'num_gpu', self.num_gpu
        yield 'num_hidden_layers', self.num_hidden_layers
        yield 'seed', self.seed
        yield 'student', self.student
        yield 'task', self.task
        yield 'type_vocab_size', self.type_vocab_size
        yield 'warmup_step', self.warmup_step
        yield 'weight_decay', self.weight_decay

    def __str__(self) -> str:
        col_width = max([
            max(len(k), len(str(v)))
            for k, v in self
        ])
        table_width = 2 * (col_width + 2) + 1
        sep = '\n+' + '-' * table_width + '+'
        row = '\n| {:<{col_width}} | {:<{col_width}} |'
        table = (
            sep +
            row.format('configuration', 'value', col_width=col_width) +
            sep +
            ''.join([
                row.format(k, v, col_width=col_width)
                for k, v in self
            ]) +
            sep
        )

        return table

    @property
    def betas(self) -> Tuple[float, float]:
        r"""Optimizer AdamW's parameter `betas`.

        Returns:
            A tuple contain two values, `self.beta1, self.beta2`.
        """
        return self.beta1, self.beta2

    @property
    def device(self) -> torch.device:
        r"""Get running model device.

        If `self.num_gpu == 0`, then run model on CPU.
        Else run model on CUDA device.

        Returns:
            Device create by `torch.device`.
        """
        if not self.num_gpu:
            return torch.device('cpu')
        return torch.device('cuda')

    def save(self) -> None:
        r"""Save configuration into json file."""
        file_path = StudentConfig.file_path(
            experiment=self.experiment,
            student=self.student,
            task=self.task
        )

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as json_file:
            json.dump(
                dict(self),
                json_file,
                ensure_ascii=False
            )

    @classmethod
    def load(
            cls,
            experiment: str = '',
            student: str = '',
            task: str = ''
    ):
        r"""Load configuration from json file.

        Args:
        experiment:
            Name of experiment.
        student:
            Student model's name.
        task:
            Name of fine-tune task.
        """
        file_path = StudentConfig.file_path(
            experiment=experiment,
            student=student,
            task=task
        )
        with open(file_path, 'r') as json_file:
            return cls(**json.load(json_file))

    @staticmethod
    def format_experiment_name(
            experiment: str = '',
            student: str = '',
            task: str = ''
    ) -> str:
        r"""Return formatted experiment name.

        Args:
            experiment:
                Name of experiment.
            student:
                Student model's name.
            task:
                Name of fine-tune task.

        Returns:
            A string of experiment name.
        """
        return f'distill-{student}-{task}-{experiment}'

    @staticmethod
    def file_path(
            experiment: str = '',
            student: str = '',
            task: str = ''
    ) -> str:
        r"""Return formatted config file name.

        Args:
            experiment:
                Name of experiment.
            student:
                Student model's name.
            task:
                Name of fine-tune task.

        Returns:
            A string of config file path.
        """
        return '{}/{}/{}'.format(
            fine_tune.path.FINE_TUNE_EXPERIMENT,
            StudentConfig.format_experiment_name(
                experiment=experiment,
                student=student,
                task=task
            ),
            fine_tune.path.CONFIG
        )
