"""Helper function for evaluating fine-tuned model.

Usage:
    fine_tune.util.eval(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

from typing import Tuple
from typing import Union

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model
import fine_tune.path
import fine_tune.util


@torch.no_grad()
def evaluation(
        config: fine_tune.config.TeacherConfig,
        dataset: Union[
            fine_tune.task.MNLI,
        ],
) -> Tuple[int, str]:
    """Evaluate fine-tuned model on task specific dataset.

    Args:
        config:
            `fine_tune.config.TeacherConfig` which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.

    Returns:
        Max accuracy and its repective checkpoint.
    """

    # Get experiment name and path.
    experiment_name = fine_tune.config.TeacherConfig.format_experiment_name(
        experiment=config.experiment,
        task=config.task,
        teacher=config.teacher
    )
    experiment_path = '{}/{}'.format(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Create tokenizer.
    tokenizer = fine_tune.util.load_tokenizer_by_config(
        config=config
    )

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(
            max_seq_len=config.max_seq_len,
            tokenizer=tokenizer
        ),
        shuffle=True
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        '{}/{}'.format(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Get all checkpoint file names.
    ckpt_pattern = r'(\d+)\.ckpt'
    all_ckpts = filter(
        lambda file_name: re.match(ckpt_pattern, file_name),
        os.listdir(experiment_path)
    )
    all_ckpts = sorted(
        all_ckpts,
        key=lambda file_name: int(re.match(ckpt_pattern, file_name).group(1))
    )

    # Record max accuracy and its respective checkpoint.
    max_acc = 0
    max_acc_ckpt = '0.ckpt'

    for ckpt in all_ckpts:
        model = fine_tune.util.load_teacher_model_by_config(
            config=config
        )
        model.load_state_dict(torch.load(
            f'{experiment_path}/{ckpt}'
        ))
        # Evaluation mode.
        model.eval()

        all_label = []
        all_pred_label = []

        # Enumerate `math.ceil(len(dataset) / config.batch_size)` times.
        # Use `torch.utils.data.DataLoader` to sample dataset.
        mini_batch_iterator = tqdm(
            dataloader,
            f'{ckpt:12}, max_acc: {max_acc:.6f}, max_ckpt: {max_acc_ckpt:10}'
        )
        for (
                input_ids,
                attention_mask,
                token_type_ids,
                label,
                _
        ) in mini_batch_iterator:

            # Mini-batch prediction.
            pred_label = model(
                input_ids=input_ids.to(config.device),
                token_type_ids=token_type_ids.to(config.device),
                attention_mask=attention_mask.to(config.device)
            ).to('cpu').argmax(dim=-1)

            all_label.extend(label.tolist())
            all_pred_label.extend(pred_label.tolist())

        # Calculate accuracy.
        acc = accuracy_score(all_label, all_pred_label)

        # Update max accuracy.
        if max_acc <= acc:
            max_acc = acc
            max_acc_ckpt = ckpt

        # Log accuracy.
        writer.add_scalar(
            f'{config.task}-{config.dataset}/accuracy',
            acc,
            int(re.match(ckpt_pattern, ckpt).group(1))
        )

    writer.flush()
    writer.close()

    return max_acc, max_acc_ckpt
