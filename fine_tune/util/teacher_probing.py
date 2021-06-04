r"""This scirpt is a temporary script for teacher model probing task.
We want to know what will happen if we remove some teacher layer
and train it again.
#TODO: Remove this temporary file.
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# 3rd party modules

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import transformers

from tqdm import tqdm

# my own modules

import fine_tune.config
import fine_tune.task
import fine_tune.model
import fine_tune.path

def train_shallower_bert(
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.Dataset,
    teacher_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    student_tokenizer: transformers.PreTrainedTokenizer
):
    """Train a thiner BERT model for probing task.
    Here we use `student_model` to refer our thiner BERT model.

    Parameters
    ----------
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    dataset : fine_tune.task.Dataset
        Task specific dataset.
    teacher_model : fine_tune.model.TeacherModel
        A fine-tuned teacher model which is used to init thiner bert weight.
    student_model : fine_tune.model.StudentModel
        Our thiner BERT model.
    optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    """
    # Set teacher model as evaluation mode.
    teacher_model.eval()

    # Set student model as training mode.
    student_model.train()

    device = student_config.device

    # Clean all gradient.
    optimizer.zero_grad()

    # Init our thiner BERT model from fine-tuned teacher.
    # ============================================================================================
    teacher_indices = [int(item)-1 for item in input("Enter desired teacher layer:\n").split()]

    print("Load fine-tuned teacher weight")

    teacher_encoder_weight = teacher_model.encoder.state_dict()
    new_state_dict = {}
    keys = [
        'attention.self.query.weight',
        'attention.self.query.bias',
        'attention.self.key.weight',
        'attention.self.key.bias',
        'attention.self.value.weight',
        'attention.self.value.bias',
        'attention.output.dense.weight',
        'attention.output.dense.bias',
        'attention.output.LayerNorm.weight',
        'attention.output.LayerNorm.bias',
        'intermediate.dense.weight',
        'intermediate.dense.bias',
        'output.dense.weight',
        'output.dense.bias',
        'output.LayerNorm.weight',
        'output.LayerNorm.bias'
    ]

    for i, t_index in enumerate(teacher_indices):
        for key in keys:
            new_state_dict.update(
                {
                    f'encoder.layer.{i}.{key}':
                    teacher_encoder_weight[f'encoder.layer.{t_index}.{key}']
                }
            )

    new_state_dict.update(
        {
            'pooler.dense.weight':teacher_encoder_weight['pooler.dense.weight'],
            'pooler.dense.bias':teacher_encoder_weight['pooler.dense.bias']
        }
    )

    student_model.encoder.load_state_dict(
        new_state_dict,
        strict=False
    )

    teacher_linear_weight = teacher_model.linear_layer.state_dict()
    student_model.linear_layer.load_state_dict(
        teacher_linear_weight
    )
    print('Finish initilization')

    # ============================================================================================

    # Get experiment name and path for student model.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=student_config.experiment,
        model=student_config.model,
        task=student_config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Construct data loader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=student_config.batch_size // student_config.accum_step,
        collate_fn=dataset.create_collate_fn(),
        num_workers=os.cpu_count(),
        shuffle=True
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Use cross-entropy as objective.
    objective = nn.CrossEntropyLoss()

    # Step and accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accumulate loss.
    # Update when accumulate to `student_config.batch_size`.
    loss = 0
    accum_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f}',
        total=student_config.total_step
    )

    # Total update times: `config.total_step`.
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for (
                text,
                text_pair,
                label
        ) in dataloader:

            # Transform `label` to tensor.
            label = torch.LongTensor(label)

            # Get `input_ids`, `token_type_ids` and `attention_mask` via tokenizer.
            batch_encode = student_tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=student_config.max_seq_len,
                return_tensors='pt',
                truncation=True
            )
            input_ids = batch_encode['input_ids']
            token_type_ids = batch_encode['token_type_ids']
            attention_mask = batch_encode['attention_mask']

            # Accumulate cross-entropy loss.
            # Use `model(...)` to do forward pass.
            logits, _ = student_model(
                    input_ids=input_ids.to(device),
                    token_type_ids=token_type_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
            accum_loss = objective(
                input=logits,
                target=label.to(device)
            ) / student_config.accum_step

            # Mini-batch cross-entropy loss. Only used as log.
            loss += accum_loss.item()

            # Backward pass accumulation loss.
            accum_loss.backward()

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch size.
            if accum_step % student_config.accum_step == 0:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    student_config.max_norm
                )

                # Gradient descend.
                optimizer.step()

                # Update learning rate.
                scheduler.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'loss: {loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `config.log_step` step.
                if step % student_config.log_step == 0:
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `config.ckpt_step` step.
                if step % student_config.ckpt_step == 0:
                    torch.save(
                        student_model.state_dict(),
                        os.path.join(experiment_dir, f'model-{step}.pt')
                    )

            # Stop training condition.
            if accum_step >= total_accum_step:
                break

    # Release IO resources.
    writer.flush()
    writer.close()
    cli_logger.close()

    # Save the lastest model.
    torch.save(
        student_model.state_dict(),
        os.path.join(experiment_dir, f'model-{step}.pt')
    )
