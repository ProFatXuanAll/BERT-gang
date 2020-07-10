r"""Helper function for generating soft-target from fine-tuned model.

Usage:
    fine_tune.util.gen_soft_target(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union

# 3rd party modules

import torch
import torch.utils
import torch.utils.data

from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model
import fine_tune.path
import fine_tune.util


@torch.no_grad()
def gen_soft_target(
        ckpt: int,
        config: fine_tune.config.TeacherConfig,
        dataset: Union[
            fine_tune.task.MNLI,
        ],
):
    r"""Generate fine-tuned model soft-target on task specific dataset.

    Args:
        ckpt:
            Which checkpoint to generate soft target.
        config:
            `fine_tune.config.TeacherConfig` which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
    """

    # Get experiment name.
    experiment_name = fine_tune.config.TeacherConfig.format_experiment_name(
        experiment=config.experiment,
        task=config.task,
        teacher=config.teacher
    )

    # Get checkpoint file path.
    ckpt_path = '{}/{}/{}.ckpt'.format(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name,
        ckpt
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
        shuffle=False
    )

    # Load model from checkpoint.
    model = fine_tune.util.load_teacher_model_by_config(
        config=config
    )
    model.load_state_dict(torch.load(ckpt_path))

    # Evaluation mode.
    model.eval()

    # Sample index for updating soft target.
    cur_sample_index = 0

    # Enumerate `math.ceil(len(dataset) / config.batch_size)` times.
    # Use `torch.utils.data.DataLoader` to sequentially sample dataset.
    for (
            input_ids,
            attention_mask,
            token_type_ids,
            _,
            _
    ) in tqdm(dataloader):

        # Mini-batch prediction.
        pred = model(
            input_ids=input_ids.to(config.device),
            token_type_ids=token_type_ids.to(config.device),
            attention_mask=attention_mask.to(config.device)
        ).to('cpu').tolist()

        # Update soft target.
        for index, soft_target in enumerate(pred):
            dataset.update_soft_target(
                index=index + cur_sample_index,
                soft_target=soft_target
            )

        # Shift sample index.
        cur_sample_index += len(pred)

    # Save soft target.
    dataset.save_for_distillation(experiment_name)
