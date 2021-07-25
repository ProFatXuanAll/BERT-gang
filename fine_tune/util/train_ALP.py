r"""Implement ALP-KD and ALP-NO training scripts.
Note: These functions may use 2 GPU device.
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

def train_alp_kd(
    teacher_config: fine_tune.config.TeacherConfig,
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.Dataset,
    teacher_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    teacher_tokenizer: transformers.PreTrainedTokenizer,
    student_tokenizer: transformers.PreTrainedTokenizer,
    soft_weight: float = 0.2,
    hard_weight: float = 0.8,
    mse_weight: int = 100,
    softmax_temp: float = 1.0
):
    """Train ALP-KD model.

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
        generate soft targets, hidden states and attentions
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
    soft_weight : float, optional
        Weight of soft target loss, by default 0.2
    hard_weight : float, optional
        Weight of hard target loss, by default 0.8
    mse_weight : int, optional
        Weight of hidden MSE loss, by default 100
    softmax_temp : float, optional
        Softmax temperature, by default 1.0
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

    skip = 12 // student_config.num_hidden_layers
    teacher_indices = list(range(skip-1, 12, skip))

    # Init student model from pre-trained teacher layer.
    student_model.init_from_pre_trained(teacher_indices = teacher_indices)
    student_model.train()

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

            # Calculate logits loss.
            batch_logits_loss = logits_objective(
                hard_target=label.to(student_device),
                teacher_logits=teacher_logits.to(student_device),
                student_logits=student_logits,
                gamma=hard_weight,
                alpha=soft_weight,
                softmax_temp=softmax_temp
            )

            # Normalize loss.
            batch_logits_loss = batch_logits_loss / student_config.accum_step

            # Log loss.
            logits_loss += batch_logits_loss.item()
            loss += batch_logits_loss.item()

            # Accumulate gradients.
            batch_logits_loss.backward(retain_graph=True)

            # Calculate hidden MSE loss.
            # Drop embedding layer
            teacher_hiddens = teacher_hiddens[1:]
            student_hiddens = student_hiddens[1:]

            # Extract each teacher layer's [CLS] emebedding.
            teacher_cls = [hidden[:,0,:] for hidden in teacher_hiddens]

            # `teacher_cls`: BxLxH
            # `B`: batch size
            # `L`: layer number
            # `H`: cls embedding dimension
            teacher_cls = torch.stack(teacher_cls).transpose(0,1)

            # This list is used for saving attention score ckpt.
            attn_score_list = []

            for s_hidden in student_hiddens:
                s_cls = s_hidden[:,0,:]
                dim = s_cls.shape[-1]
                s_cls = torch.unsqueeze(s_cls,-1)

                # Compute dot product.
                # `attn_score`: BxL.
                inner_prod = teacher_cls @ s_cls / torch.sqrt(torch.as_tensor(dim))
                inner_prod = torch.squeeze(inner_prod, dim=-1)

                attn_score = torch.exp(
                    inner_prod
                )

                attn_score = attn_score / (
                    torch.sum(attn_score,dim=-1,keepdim=True)
                )

                # Compute aggregated hidden states.
                # `attn_score`: BxLx1
                attn_score = torch.unsqueeze(attn_score, dim=-1)

                attn_score_list.append(attn_score)

                # `agg_hidden`: BxH
                agg_hidden = teacher_cls * attn_score # BxLxH
                agg_hidden = torch.sum(agg_hidden, dim=1)

                # reshape `s_cls` to BxH
                s_cls = torch.squeeze(s_cls, dim=-1)

                batch_hidden_loss = hidden_objective(
                    teacher_hidden=agg_hidden,
                    student_hidden=s_cls,
                    mu=mse_weight
                ) / student_config.num_hidden_layers

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

                    for layer, score in enumerate(attn_score_list):
                        torch.save(
                            score,
                            os.path.join(experiment_dir, f'attn-{layer}-{step}.pt')
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
    for layer, score in enumerate(attn_score_list):
        torch.save(
            score,
            os.path.join(experiment_dir, f'attn-{layer}-{step}.pt')
        )

def train_alp_kd_hidden(
    teacher_config: fine_tune.config.TeacherConfig,
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.Dataset,
    teacher_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    teacher_tokenizer: transformers.PreTrainedTokenizer,
    student_tokenizer: transformers.PreTrainedTokenizer,
    soft_weight: float = 0.2,
    hard_weight: float = 0.8,
    mse_weight: int = 100,
    softmax_temp: float = 1.0
):
    """Train ALP-KD but we use teacher model's hidden states
    as our learning objective.

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
        generate soft targets, hidden states and attentions
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
    soft_weight : float, optional
        Weight of soft target loss, by default 0.2
    hard_weight : float, optional
        Weight of hard target loss, by default 0.8
    mse_weight : int, optional
        Weight of hidden MSE loss, by default 100
    softmax_temp : float, optional
        Softmax temperature, by default 1.0
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

    skip = 12 // student_config.num_hidden_layers
    teacher_indices = list(range(skip-1, 12, skip))

    # Init student model from pre-trained teacher layer.
    student_model.init_from_pre_trained(teacher_indices = teacher_indices)
    student_model.train()

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

            # Calculate logits loss.
            batch_logits_loss = logits_objective(
                hard_target=label.to(student_device),
                teacher_logits=teacher_logits.to(student_device),
                student_logits=student_logits,
                gamma=hard_weight,
                alpha=soft_weight,
                softmax_temp=softmax_temp
            )

            # Normalize loss.
            batch_logits_loss = batch_logits_loss / student_config.accum_step

            # Log loss.
            logits_loss += batch_logits_loss.item()
            loss += batch_logits_loss.item()

            # Accumulate gradients.
            batch_logits_loss.backward(retain_graph=True)

            # Calculate hidden MSE loss.
            # Drop embedding layer
            teacher_hiddens = teacher_hiddens[1:]
            student_hiddens = student_hiddens[1:]

            # Stack all teacher layer hidden states
            # `all_teacher_hiddens`: BxSxLxH
            # B: batch size, S: sequence length
            # L: layer numbers H: hidden states dimension
            all_teacher_hiddens = torch.stack(teacher_hiddens).transpose(0,1)
            # BxLxSxH
            all_teacher_hiddens = all_teacher_hiddens.transpose(1,2)
            # BxSxLxH

            # `attn_score_list` is used for storing attention scroe matrix.
            attn_score_list = []

            # Calculate attention score of each student layer.
            for s_hidden in student_hiddens:
                dim = s_hidden.shape[-1]

                # `s_hidden`: BxSxHx1
                s_hidden = torch.unsqueeze(s_hidden, -1)

                # Compute inner product.
                # `inner_prod`: BxSxL
                inner_prod = all_teacher_hiddens @ s_hidden / torch.sqrt(torch.as_tensor(dim))
                inner_prod = torch.squeeze(inner_prod, dim=-1)

                # Compute attention score.
                # `attn_score`: BxSxL
                attn_score = torch.exp(inner_prod)
                attn_score = attn_score / (
                    torch.sum(attn_score, dim=-1, keepdim=True)
                )

                # Compute aggregated hidden states.
                # `attn_score`: BxSx1xL
                attn_score = torch.unsqueeze(attn_score, dim=-1)
                attn_score_list.append(attn_score)

                # `agg_hidden`: BxSxH
                agg_hidden = all_teacher_hiddens * attn_score # BxSxLxH
                agg_hidden = torch.sum(agg_hidden, dim=-2)

                # reshape `s_hidden` to BxSxH
                s_hidden = torch.squeeze(s_hidden, dim=-1)

                # Compute MSE loss.
                batch_hidden_loss = hidden_objective(
                    teacher_hidden=agg_hidden,
                    student_hidden=s_hidden,
                    mu=mse_weight
                ) / student_config.num_hidden_layers

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

                    for layer, score in enumerate(attn_score_list):
                        torch.save(
                            score,
                            os.path.join(experiment_dir, f'attn-{layer}-{step}.pt')
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

    for layer, score in enumerate(attn_score_list):
        torch.save(
            score,
            os.path.join(experiment_dir, f'attn-{layer}-{step}.pt')
        )
