#TODO: replace `amp_distill` with this file.
r"""Helper functions for knowledge distillation with automatic mixed precision.
Note: This functions use 2 GPU device to perform distillation.
Usage:
    import fine_tune

    fine_tune.util.amp_distill(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# 3rd party modules

import torch
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


def amp_distill_mgpu(
        teacher_config: fine_tune.config.TeacherConfig,
        student_config: fine_tune.config.StudentConfig,
        dataset: fine_tune.task.Dataset,
        teahcer_model: fine_tune.model.TeacherModel,
        student_model: fine_tune.model.StudentModel,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        teacher_tokenizer: transformers.PreTrainedTokenizer,
        student_tokenizer: transformers.PreTrainedTokenizer
):
    r"""Perform knowledge distillation from given fine-tuned teacher model
    with automatic mixed precision.
    Note: This function will use two gpu-device,
    you should check it with `fine_tune.util.check_device` before calling this function.

    Args:
        teacher_config:
            `fine_tune.config.TeacherConfig` class which attributes are used
            for experiment setup.
        student_config:
            `fine_tune.config.StudentConfig` class which attributes are used
            for experiment setup.
        dataset:
            Task specific dataset.
        teacher_model:
            A fine-tuned teacher model which is used to
            generate soft targets, hidden states and attentions.
        student_model:
            Model which will perform disitllation according to
            outputs from given teacher model.
        optimizer:
            `torch.optim.AdamW` optimizer.
        schduler:
            Linear warmup scheduler provided by `transformers` package.
        teacher_tokenizer:
            Tokenizer paired with `teacher_model`.
        student_tokenizer:
            Tokenizer paired with `student_model`.
    """
    # Create a GradScalaer.
    scaler = torch.cuda.amp.GradScaler()

    # Set teacher model as evaluation mode.
    teahcer_model.eval()

    # Set student model as training mode.
    student_model.train()

    # Model running device of teacher and student model.
    teacher_device = teacher_config.device
    student_device = student_config.device

    # Clean all gradient.
    optimizer.zero_grad()

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

    # Create dataloader of teacher and student model.
    teacher_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=teacher_config.batch_size // teacher_config.accum_step,
        collate_fn=dataset.create_collate_fn(
            max_seq_len=teacher_config.max_seq_len,
            tokenizer=teacher_tokenizer
        ),
        shuffle=True
    )

    student_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=student_config.batch_size // student_config.accum_step,
        collate_fn=dataset.create_collate_fn(
            max_seq_len=student_config.max_seq_len,
            tokenizer=student_tokenizer
        ),
        shuffle=True
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Create objective functions.
    logits_objective = fine_tune.objective.distill_loss
    attn_objective =fine_tune.objective.attention_KL_loss
    hidden_objective = fine_tune.objective.hidden_MSE_loss

    # Create adaptive layer.
    # Transform dimension of student hidden states as teacher's.
    adaptive_layer = torch.nn.Linear(
        in_features=student_config.d_model,
        out_features=teahcer_model.allow_ptrain_ver[
            teacher_config.ptrain_ver
        ]
    )

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    loss = 0
    batch_logits_loss = 0
    batch_hidden_loss = 0
    batch_attn_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f}',
        total=student_config.total_step
    )

    # Total update times: `student_config.total_step`
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for (
                teacher_input_ids,
                teacher_attention_mask,
                teacher_token_type_ids,
                _
        ), (
                student_input_ids,
                student_attention_mask,
                student_token_type_ids,
                label
        ) in zip(teacher_dataloader, student_dataloader):

            # Get output logits, hidden states and attentions from teacher.
            with torch.no_grad():
                teacher_logits, teacher_hiddens, teacher_attns = teahcer_model(
                    input_ids = teacher_input_ids.to(teacher_device),
                    token_type_ids=teacher_token_type_ids.to(teacher_device),
                    attention_mask=teacher_attention_mask.to(teacher_device),
                    return_hidden_and_attn=True
                )

            # Calculate logits loss.
            # Cause parameter update in Mixed Precision Training use 32-bit fp.
            # We need to leave context manager before `backward`.

            # Enable autocast.
            with torch.cuda.amp.autocast():
                # Get output logits, hidden states and attentions from student.
                student_logits, student_hiddens, student_attns = student_model(
                    input_ids = student_input_ids.to(student_device),
                    token_type_ids=student_token_type_ids.to(student_device),
                    attention_mask=student_attention_mask.to(student_device),
                    return_hidden_and_attn=True
                )

                # Calculate batch loss of logits.
                batch_logits_loss = logits_objective(
                    hard_target=label.to(student_device),
                    teacher_logits=teacher_logits.to(student_device),
                    student_logits=student_logits
                )

                # Normalize loss.
                batch_logits_loss = batch_logits_loss / student_config.accum_step

            # Log loss.
            loss += batch_logits_loss.item()

            # Accumulate gradients.
            scaler.scale(batch_logits_loss).backward()

            # Calculate batch hidden states loss.
            # Cause parameter update in Mixed Precision Training use 32-bit fp.
            # We need to leave context manager before `backward`.
            # TODO: can we use `map` function to summarize loss?
            skip = (len(teacher_hiddens) - 1) // (len(student_hiddens) - 1)
            for t_hidden, s_hidden in zip(
                teacher_hiddens[1::skip],
                student_hiddens[1:]
            ):

                # Enable autocast.
                with torch.cuda.amp.autocast():
                    batch_hidden_loss = hidden_objective(
                        teacher_hidden=t_hidden.to(student_device),
                        student_hidden= adaptive_layer(
                            s_hidden
                        )
                    )

                    # Normalize loss.
                    batch_hidden_loss = batch_hidden_loss / student_config.accum_step

                # Log loss.
                loss += batch_hidden_loss.item()

                # Accumulate gradient.
                scaler.scale(batch_hidden_loss).backward()

            # Calculate batch attentions loss.
            # Cause parameter update in Mixed Precision Training use 32-bit fp.
            # We need to leave context manager before `backward`.
            # TODO: can we use `map` function to summarize loss?
            skip = len(teacher_attns) // len(student_attns)
            for t_attn, s_attn in zip(
                teacher_attns[skip-1::skip],
                student_attns
            ):

                # Enable autocast.
                with torch.cuda.amp.autocast():
                    batch_attn_loss = attn_objective(
                        teacher_attn=t_attn.to(student_device),
                        student_attn=s_attn
                    )

                    # Normalize loss.
                    batch_attn_loss = batch_attn_loss / student_config.accum_step

                # Log loss.
                loss += batch_attn_loss.item()

                # Accumulate gradient.
                scaler.scale(batch_attn_loss).backward()

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch size.
            if accum_step % student_config.accum_step == 0:
                # Unscale the gradient by optimizer.
                scaler.unscale_(optimizer)

                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    student_config.max_norm
                )

                # Gradient descend.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # Update learning rate.
                scheduler.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'loss: {loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `student_config.log_step`.
                if step % student_config.log_step == 0:
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `student_config.ckpt_step` step.
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

    # Save the latest model.
    torch.save(
        student_model.state_dict(),
        os.path.join(experiment_dir, f'model-{step}.pt')
    )
