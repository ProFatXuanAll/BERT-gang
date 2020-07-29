r"""Helper functions for generating logits from fine-tuned model.

Usage:
    import fine_tune

    fine_tune.util.gen_logits(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import transformers

from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model
import fine_tune.path
import fine_tune.util


@torch.no_grad()
def gen_logits(
        config: fine_tune.config.BaseConfig,
        dataset: fine_tune.task.Dataset,
        model: fine_tune.model.Model,
        tokenizer: transformers.PreTrainedTokenizer
):
    r"""Generate fine-tuned model logits on task specific dataset.

    Args:
        config:
            `fine_tune.config.BaseConfig` subclass which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        model:
            Model which will generate logits on `dataset`.
        tokenizer:
            Tokenizer paired with `model`.
    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

    # Get experiment name and model name.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
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

    # Sample index for updating logits.
    sample_index = 0

    # Generate logits through mini-batch loop.
    for (
            input_ids,
            attention_mask,
            token_type_ids,
            _,
            _
    ) in tqdm(dataloader):

        # Get mini-batch logits.
        batch_logits = model(
            input_ids=input_ids.to(device),
            token_type_ids=token_type_ids.to(device),
            attention_mask=attention_mask.to(device)
        ).to('cpu').tolist()

        # Update logits.
        for index, logits in enumerate(batch_logits):
            dataset.update_logits(
                index=index + sample_index,
                logits=logits
            )

        # Shift sample index.
        sample_index += len(batch_logits)

    # Save logits.
    dataset.save_for_distill(experiment_name)
