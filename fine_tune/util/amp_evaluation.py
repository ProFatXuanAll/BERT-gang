r"""Helper functions for evaluating fine-tuned model with automatic mixed precision.

Usage:
    import fine_tune

    acc = fine_tune.util.amp_evaluation(...)
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
import torch.utils.tensorboard
import transformers

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model


@torch.no_grad()
def amp_evaluation(
        config: fine_tune.config.BaseConfig,
        dataset: fine_tune.task.Dataset,
        model: fine_tune.model.Model,
        tokenizer: transformers.PreTrainedTokenizer
) -> float:
    r"""Evaluate model on task specific dataset with automatic mixed precision.
    Args:
        config:
            `fine_tune.config.BaseConfig` subclass which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        model:
            Model which will be evaluated on `dataset`.
        tokenizer:
            Tokenizer paired with `model`.

    Returns:
        Accuracy.

    """
    # Evaluation mode.
    model.eval()

    # Model running device.
    device = config.device

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

    # Record label and prediction for calculating accuracy.
    all_label = []
    all_pred_label = []

    # Evaluate through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    for (
            input_ids,
            attention_mask,
            token_type_ids,
            label
    ) in mini_batch_iterator:
        # Enable autocast
        with torch.cuda.amp.autocast():
            # Mini-batch prediction.
            pred_label = model.predict(
                input_ids=input_ids.to(device),
                token_type_ids=token_type_ids.to(device),
                attention_mask=attention_mask.to(device)
            ).argmax(dim=-1).to('cpu')

        all_label.extend(label.tolist())
        all_pred_label.extend(pred_label.tolist())

    # Calculate accuracy.
    acc = accuracy_score(all_label, all_pred_label)

    # Show accuracy.
    mini_batch_iterator.set_description(f'accuracy: {acc:.6f}')

    # Release IO resources.
    mini_batch_iterator.close()

    return acc
