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
        gamma: float = 0.8,
        mu: int = 100,
        softmax_temp: float = 1.0
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
    gamma: float, optional
        Weight of hard target loss, by default 0.8
    mu : int, optional
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

    # Create layer mapping indices.
    skip = 12 // student_config.num_hidden_layers
    teacher_indices = list(range(skip-1, 12, skip))

    # Init student model from pre-trained teacher layer.
    #TODO: remove warning.
    print("Warning!: Use even layer of teacher model to init student.")
    student_model.init_from_pre_trained(teacher_indices)


    hidden_objective = fine_tune.objective.hidden_MSE_loss

    #TODO: refactor
    print("Now use 12 gate to aggregate all layer hidden states")
    gate_networks = [ fine_tune.model.HighwayGate(dimension=768).to(student_device)
        for _ in range(12)
    ]

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

            # Calculate classification loss.
            # Calculate batch loss of logits.
            batch_logits_loss = logits_objective(
                hard_target=label.to(student_device),
                teacher_logits=teacher_logits.to(student_device),
                student_logits=student_logits,
                gamma=gamma,
                alpha=alpha,
                softmax_temp=softmax_temp
            )

            # Normalize loss.
            batch_logits_loss = batch_logits_loss / student_config.accum_step

            # Log loss.
            logits_loss += batch_logits_loss.item()
            loss += batch_logits_loss.item()

            # Accumulate gradients.
            batch_logits_loss.backward(retain_graph=True)

            # Calculate batch hidden states loss.
            # `teacher_hiddens`: tuple(Emb, Ht_1, Ht_2,...,Ht_12)
            # `student_hiddens`: tuple(Emb, Hs_1, Hs_2,...,Hs_6)
            # Mapping strategy:
            # Ht_2 -> Hs_1, Ht_4 -> Hs_2,...,Ht_12 -> Hs_6

            # Drop embedding layer
            teacher_hiddens = teacher_hiddens[1:]
            student_hiddens = student_hiddens[1:]

            aggregate_hiddens = []
            prev = torch.zeros_like(teacher_hiddens[0])
            # Construct aggregate hidden states
            for t_hidden, gate in zip(teacher_hiddens, gate_networks):
                agg_hidden = gate(
                    input1=prev.to(student_device),
                    input2=t_hidden.to(student_device)
                )
                aggregate_hiddens.append(agg_hidden)
                prev = agg_hidden

            for t_index, s_hidden in zip(
                    teacher_indices,
                    student_hiddens
                ):

                batch_hidden_loss = hidden_objective(
                    teacher_hidden=aggregate_hiddens[t_index].to(student_device),
                    student_hidden=s_hidden,
                    mu=mu
                ) / 6

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

                    for i, gate in enumerate(gate_networks):
                        torch.save(
                            gate.state_dict(),
                            os.path.join(experiment_dir, f'gate-{i}-{step}.pt')
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
