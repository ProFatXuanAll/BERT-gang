r"""Helper functions for reversed knowledge distillation.
Usage:
    import fine_tune

    fine_tune.util.reversed_KD(...)
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


def reversed_KD(
    teacher_config: fine_tune.config.TeacherConfig,
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.Dataset,
    teacher_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    teacher_tokenizer: transformers.PreTrainedTokenizer,
    student_tokenizer: transformers.PreTrainedTokenizer,
    alpha: float = 0.2,
    softmax_temp: float = 1.0
):
    """Peform reverse knowledge distillation

    Parameters
    ----------
    teacher_config : fine_tune.config.TeacherConfig
        `fine_tune.config.TeacherConfig` class which attributes are used
        for experiment setup.
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    dataset : fine_tune.task.Dataset
        Task specific dataset.
    teacher_model : fine_tune.model.TeacherModel
        A fine-tuned teacher model, ready to be trained by student.
    student_model : fine_tune.model.StudentModel
        A well-trained student model, ready to transfer knowledge to teacher.
    optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    teacher_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `teacher_model`.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    alpha: float, optional
        Weight of soft target cross entropy, default 0.2
    softmax_temp: float, optional
        Softmax temperature of distillation cross entropy loss, default 1.0
    """

    # Set student model as evaluation mode.
    student_model.eval()

    # Set teacher model as training mode.
    teacher_model.train()

    # Model running device of teacher and student model.
    teacher_device = teacher_config.device
    student_device = student_config.device

    # Clean all gradient.
    optimizer.zero_grad()

    # Get experiment name and path for new teacher model.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=teacher_config.experiment,
        model=teacher_config.model,
        task=teacher_config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Teacher and student share a dataloader.
    # Teacher and student share a dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=teacher_config.batch_size // teacher_config.accum_step,
        collate_fn=dataset.create_collate_fn(),
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

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = teacher_config.total_step * teacher_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    loss = 0
    # logits_loss = 0

    # torch.Tensor placeholder.
    batch_logits_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f}',
        total=teacher_config.total_step
    )

    # Total update times: `teacher_config.total_step`
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for text, text_pair, label in dataloader:
            # Transform `label` to a Tensor.
            label = torch.LongTensor(label)

            # Get `input_ids`, `token_type_ids` and `attention_mask` from via tokenizer.
            teacher_batch_encode = teacher_tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=teacher_config.max_seq_len,
                return_tensors='pt',
                truncation=True
            )
            teacher_input_ids = teacher_batch_encode['input_ids']
            teacher_token_type_ids = teacher_batch_encode['token_type_ids']
            teacher_attention_mask = teacher_batch_encode['attention_mask']

            student_batch_encode = student_tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=student_config.max_seq_len,
                return_tensors='pt',
                truncation=True
            )
            student_input_ids = student_batch_encode['input_ids']
            student_token_type_ids = student_batch_encode['token_type_ids']
            student_attention_mask = student_batch_encode['attention_mask']

            # Get output logits from student model.
            with torch.no_grad():
                student_logits = student_model(
                    input_ids = student_input_ids.to(student_device),
                    token_type_ids = student_token_type_ids.to(student_device),
                    attention_mask = student_attention_mask.to(student_device)
                )

            # Get output logits from teacher model.
            teacher_logits = student_model(
                input_ids = teacher_input_ids.to(teacher_device),
                token_type_ids = teacher_token_type_ids.to(teacher_device),
                attention_mask = teacher_attention_mask.to(teacher_device)
            )

            # Calculate logits loss.
            batch_logits_loss = logits_objective(
                hard_target=label.to(teacher_device),
                teacher_logits=student_logits.to(teacher_device),
                student_logits=teacher_logits,
                alpha=alpha,
                softmax_temp=softmax_temp
            )
            # Normalize loss.
            batch_logits_loss = batch_logits_loss / teacher_config.accum_step

            # Log loss.
            loss += batch_logits_loss.item()

            # Accumulate gradients.
            batch_logits_loss.backward()

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch size.
            if accum_step % teacher_config.accum_step == 0:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    teacher_model.parameters(),
                    teacher_config.max_norm
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

                # Log loss and learning for each `teacher_config.log_step`.
                if step % teacher_config.log_step == 0:
                    writer.add_scalar(
                        f'{teacher_config.task}/{teacher_config.dataset}/{teacher_config.model}'+
                        '/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{teacher_config.task}/{teacher_config.dataset}/{teacher_config.model}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `teacher_config.ckpt_step`.
                if step % teacher_config.ckpt_step == 0:
                    torch.save(
                        teacher_model.state_dict(),
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
        teacher_model.state_dict(),
        os.path.join(experiment_dir, f'model-{step}.pt')
    )
