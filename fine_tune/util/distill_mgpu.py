r"""Helper functions for knowledge distillation.
Note: This functions use 2 GPU device to perform distillation.
Usage:
    import fine_tune

    fine_tune.util.distill_mgpu(...)
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


def distill_mgpu(
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
        mu: int = 100,
        softmax_temp: float = 1.0,
        use_classify_loss: bool = True,
        use_hidden_loss: bool = True
):
    """Perform knowledge distillation from given fine-tuned teacher model
    without automatic mixed precision.
    Note: This function may use two gpu-device.

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
        A fine-tuned teacher model which is used to
        generate soft targets, hidden states and attentions.
    student_model : fine_tune.model.StudentModel
        Model which will perform disitllation according to
        outputs from given teacher model.
    optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    teacher_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `teacher_model`.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    alpha : float, optional
        Weight of soft target loss, by default 0.2
    mu : int, optional
        Weight of hidden MSE loss, by default 100
    softmax_temp : float, optional
        Softmax temperature, by default 1.0
    use_classify_loss : bool, optional
        Total loss function include classification loss which is defined by:
            L_{classify} =
                ce_weights * ( alpha * CE_{soft} + ( 1-alpha ) * CE_{hard} ) +
                (1-ce_weights) * L_{SCL}
        by default True
    use_hidden_loss : bool, optional
        Total loss function include hidden states loss, by default True
    """

    # Set teacher model as evaluation mode.
    teacher_model.eval()

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

    # Construct data loader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=teacher_config.batch_size // teacher_config.accum_step,
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

    # Create objective functions.
    logits_objective = fine_tune.objective.distill_loss
    hidden_objective = fine_tune.objective.hidden_MSE_loss

    # # TODO: Refactor
    # normalize = True
    # print("Normalize CLS embedding!")

    # TODO: Restore this block to use adaptive layer
    # Create adaptive layer.
    # Transform dimension of student hidden states as teacher's.
    # adaptvive_layers = []
    # for _ in range(student_config.num_hidden_layers):
    #     adaptvive_layers.append(
    #         torch.nn.Linear(
    #             in_features=student_config.d_model,
    #             out_features=teacher_model.allow_ptrain_ver[
    #                 teacher_config.ptrain_ver
    #             ]
    #         ).to(student_config.device_id)
    #     )

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    loss = 0
    logits_loss = 0
    hidden_loss = 0

    # torch.Tensor placeholder.
    batch_logits_loss = 0
    batch_hidden_loss = 0


    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f} ' +
            f'logits_loss: {logits_loss:.6f} ' +
            f'hidden_loss: {hidden_loss:.6f} ',
        total=student_config.total_step
    )

    # Total update times: `student_config.total_step`
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for text, text_pair, label in dataloader:

            # Transform `label` to a Tensor.
            label = torch.LongTensor(label)

            # Get `input_ids`, `token_type_ids` and `attention_mask` via tokenizer.
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

            # Get output logits, hidden states and attentions from teacher and student.
            with torch.no_grad():
                teacher_logits, teacher_hiddens, _ = teacher_model(
                    input_ids = teacher_input_ids.to(teacher_device),
                    token_type_ids=teacher_token_type_ids.to(teacher_device),
                    attention_mask=teacher_attention_mask.to(teacher_device),
                    return_hidden_and_attn=True
                )

            # Get output logits, hidden states and attentions from student.
            student_logits, student_hiddens, _ = student_model(
                input_ids = student_input_ids.to(student_device),
                token_type_ids=student_token_type_ids.to(student_device),
                attention_mask=student_attention_mask.to(student_device),
                return_hidden_and_attn=True
            )

            # Calculate classify loss.

            if use_classify_loss:
                # Calculate batch loss of logits.
                batch_logits_loss = logits_objective(
                    hard_target=label.to(student_device),
                    teacher_logits=teacher_logits.to(student_device),
                    student_logits=student_logits,
                    alpha = alpha,
                    softmax_temp = softmax_temp
                )
                # Normalize loss.
                batch_logits_loss = batch_logits_loss / student_config.accum_step

                # Log loss.
                logits_loss += batch_logits_loss.item()
                loss += batch_logits_loss.item()

                # Accumulate gradients.
                batch_logits_loss.backward(retain_graph=True)

            if use_hidden_loss:
                # Calculate batch hidden states loss.
                skip = (len(teacher_hiddens) - 1) // (len(student_hiddens) - 1)
                for t_hidden, s_hidden in zip(teacher_hiddens[1::skip],student_hiddens[1:]):
                    batch_hidden_loss = hidden_objective(
                        teacher_hidden=t_hidden.to(student_device),
                        student_hidden= s_hidden,
                        mu=mu
                    )

                    # Normalize loss.
                    batch_hidden_loss = batch_hidden_loss / student_config.accum_step

                    # Log loss.
                    hidden_loss += batch_hidden_loss.item()
                    loss += batch_hidden_loss.item()

                    # Accumulate gradient.
                    batch_hidden_loss.backward(retain_graph=True)

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
                    f'loss: {loss:.6f} ' +
                    f'logits_loss: {logits_loss:.6f} ' +
                    f'hidden_loss: {hidden_loss:.6f} ',
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `student_config.log_step`.
                if step % student_config.log_step == 0:
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
                        '/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
                        '/logits_loss',
                        logits_loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
                        '/hidden_loss',
                        hidden_loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0
                logits_loss = 0
                hidden_loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `student_config.ckpt_step` step.
                if step % student_config.ckpt_step == 0:
                    torch.save(
                        student_model.state_dict(),
                        os.path.join(experiment_dir, f'model-{step}.pt')
                    )

                    # TODO: if use adaptive layer we have to save it.

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
