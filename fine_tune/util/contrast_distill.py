r"""Helper function for knowledge distillation with contrastive learning.
Note: This functions use 2 GPU device to perform distillation.
Usage:
    import fine_tune

    fine_tune.util.contrast_distill(...)
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
import fine_tune.contrast_util


def contrast_distill(
    teacher_config: fine_tune.config.TeacherConfig,
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.ContrastDataset,
    teacher_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    teacher_tokenizer: transformers.PreTrainedTokenizer,
    student_tokenizer: transformers.PreTrainedTokenizer,
    membank: fine_tune.contrast_util.Memorybank,
    sotfmax_temp: float = 0.07,
    use_logits_loss: bool = True
):
    """Perform knowledge distillation with contrastive loss
    from given fine-tuned teacher model without automatic mixed precision.
    Note: This function will use two gpu-device.

    Parameters
    ----------
    teacher_config : fine_tune.config.TeacherConfig
        `fine_tune.config.TeacherConfig` class which attributes are used
        for experiment setup.
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    dataset : fine_tune.task.ContrastDataset
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
    membank : fine_tune.contrast_util.Memorybank
        Memory bank for contrastive learning.
    softmax_temp : float, optional
        Softmax temperature.
    use_logits_loss : bool, optional
        Total loss function include hard target and soft target logits loss, by default True
    """

    # Set teacher model to evalutation mode.
    teacher_model.eval()

    # Set student model as training mode.
    student_model.train()

    # Model running device of teacher and student model.
    teacher_device = teacher_config.device
    student_device = student_config.device

    # Memory bank device
    memdevice = membank.device

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
    contrastive_objective = nn.CrossEntropyLoss()

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    loss = 0
    logits_loss = 0
    contrast_loss = 0

    # torch.Tensor placeholder.
    batch_logits_loss = 0
    batch_contrast_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f} ' +
            f'logits_loss: {logits_loss:.6f} ' +
            f'contrast_loss: {contrast_loss:.6f}',
        total=student_config.total_step
    )

    # Total update times: `student_config.total_step`
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for text, text_pair, label, _, n_indices in dataloader:

            # Transform `label` and `n_indices` to tensor.
            label = torch.LongTensor(label)
            n_indices = torch.LongTensor(n_indices).to(memdevice)

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

            # Get output logits, [CLS] hidden from teacher.
            with torch.no_grad():
                teacher_logits, teacher_pooler = teacher_model(
                    input_ids = teacher_input_ids.to(teacher_device),
                    token_type_ids=teacher_token_type_ids.to(teacher_device),
                    attention_mask=teacher_attention_mask.to(teacher_device)
                )

            # Get output logits, [CLS] hidden from student.
            student_logits, student_pooler = student_model(
                input_ids = student_input_ids.to(student_device),
                token_type_ids=student_token_type_ids.to(student_device),
                attention_mask=student_attention_mask.to(student_device)
            )

            # Calculate logits loss.
            if use_logits_loss:
                # Calculate batch loss of logits.
                # Calculate batch loss of logits.
                batch_logits_loss = logits_objective(
                    hard_target=label.to(student_device),
                    teacher_logits=teacher_logits.to(student_device),
                    student_logits=student_logits
                )
                # Normalize loss.
                batch_logits_loss = batch_logits_loss / student_config.accum_step

                # Log loss.
                logits_loss += batch_logits_loss.item()
                loss += batch_logits_loss.item()

                # Accumulate gradients.
                batch_logits_loss.backward(retain_graph=True)

            # Calculate contrastive loss of last [CLS] hidden.
            # `q`: student last [CLS] hidden state.
            # `k+`: teacher last [CLS] hidden state.
            # `k-(negatives)`: Sample from memory bank.
            # B: batch size.
            # D: dimension.
            # K: number of negative samples.

            # Get hidden batch size and dim.
            B = teacher_pooler.shape[0]
            D = teacher_pooler.shape[1]

            # Normalize `Query` and `K+`.
            q = nn.functional.normalize(student_pooler)
            k_pos = nn.functional.normalize(teacher_pooler).to(student_device)

            # Compute logits of positive pairs.
            # `pos_logits`: tensor of shape Bx1x1
            pos_logits = torch.bmm(q.view(B, 1, D), k_pos.view(B, D, 1))

            # `pos_logits`: tensor of shape Bx1
            pos_logits = pos_logits.view(B,-1)

            # Init a temporary list to store all negative logits.
            neg_logits = []

            # Compute relative negative logit.
            for qi, idx in zip(q, n_indices):
                # `qi`: D shape tensor, represent a single query.
                # `indx`: K shape tensor, represent relative negative index.
                # Sample negative paris w.r.t a single query.
                #TODO: move membank to 'student_device'
                # k_neg = membank(idx)
                k_neg = membank(idx).to(student_device)

                # `k_neg` is a tensor of shape: DxK.
                # Compute negative logit w.r.t `qi`.
                # `n_logit` is a tensor of shape K.
                neg_logits.append(qi @ k_neg)

            # Convert to tensor of shape BxK.
            neg_logits = torch.stack(neg_logits)

            # Get `output` tensor of shape Bx(1+K).
            output = torch.cat([pos_logits, neg_logits], dim=1)

            # Apply temperature.
            output /= sotfmax_temp

            # Construct labels: positive key indicators.
            targets = torch.zeros(output.shape[0], dtype=torch.long).to(student_device)

            # Compute contrastive loss.
            batch_contrast_loss = contrastive_objective(output, targets)

            # Normalize loss.
            batch_contrast_loss = batch_contrast_loss / student_config.accum_step

            # Log loss.
            contrast_loss += batch_contrast_loss.item()
            loss += batch_contrast_loss.item()

            # Accumulate gradient.
            batch_contrast_loss.backward(retain_graph=True)

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
                    f'contrast_loss: {contrast_loss:.6f}'
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
                        '/contrast_loss',
                        contrast_loss,
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
                contrast_loss = 0

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
