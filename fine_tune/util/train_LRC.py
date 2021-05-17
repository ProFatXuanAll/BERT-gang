r"""Training script of LRC bert.
Notes
----------
[1] Fu, H., Zhou, S., Yang, Q., Tang, J., Liu, G., Liu, K., &
Li, X. (2020). LRC-BERT: Latent-representation Contrastive Knowledge
Distillation for Natural Language Understanding.
arXiv preprint arXiv:2012.07335.
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
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

def train_lrc_layerwise(
    teacher_logitsbank: fine_tune.model.Logitsbank,
    teacher_device: torch.cuda.device,
    student_config: fine_tune.config.StudentConfig,
    dataset: fine_tune.task.ContrastDataset,
    student_model: fine_tune.model.StudentModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    student_tokenizer: transformers.PreTrainedTokenizer,
    membanks: List[fine_tune.contrast_util.Memorybank],
    softmax_temp: float = 1,
    nce_cos_weight: float = 1,
    soft_weight: float = 1,
    hard_weight: float = 1,
    contrast_steps: int = 0
):
    """Training script of LRC BERT

    Parameters
    ----------
    teacher_logitsbank : fine_tune.model.Logitsbank
        `fine_tune.model.Logitsbank` class which store teacher logits outputs.
    teacher_device : torch.cuda.device
        Device of teacher logitsbank
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    dataset : fine_tune.task.ContrastDataset
        Task specific dataset.
    student_model : fine_tune.model.StudentModel
        Model which will perform disitllation according to
        outputs from given teacher model.
    optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    membanks : List[fine_tune.contrast_util.Memorybank]
        A list of memory banks for contrastive learning.
    softmax_temp : float, optional
        Softmax temperature, by default 1
    nce_cos_weight : float, optional
        Weight of NCE_COS loss, by default 1
    soft_weight : float, optional
        Weight of soft target loss, by default 1
    hard_weight : float, optional
        Weight of hard target loss, by default 1
    contrast_steps : int, optional
        Training iterations for contrastive loss only
        Set this greater than zero means two-stage trainig, by default 0
    """

    # Set student model as training mode.
    student_model.train()

    # Model running device of student model.
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

    # Teacher and student share a dataloader.
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

    # Create objective functions.
    logits_objective = fine_tune.objective.distill_loss

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step
    total_contrast_step = contrast_steps * student_config.accum_step

    # Set two stage training.
    if total_contrast_step > 0:
        use_logit_loss = False
    else:
        use_logit_loss = True

    # Mini-batch loss and accumulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    loss = 0
    logits_loss = 0
    nce_cos_loss = 0

    # Adaptive layer.
    adaptive_layers = []
    for _ in range(student_config.num_hidden_layers):
        adaptive_layers.append(
            nn.Linear(
                in_features=student_config.d_model,
                out_features=768
            ).to(student_device)
        )

    # torch.Tensor placeholder.
    batch_logits_loss = 0
    batch_nce_cos_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f} ' +
            f'logits_loss: {logits_loss:.6f} ' +
            f'nce_cos_loss: {nce_cos_loss:.6f} ',
        total=student_config.total_step
    )

    # Total update times: `student_config.total_step`
    while accum_step < total_accum_step:
        # Mini-batch loop.
        for text, text_pair, label, p_indices, n_indices in dataloader:
            label = torch.LongTensor(label)
            p_indices = torch.LongTensor(p_indices)

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

            # Get ouput logits, hidden states of all layers from student.
            student_logits, student_hiddens, _ = student_model(
                input_ids = student_input_ids.to(student_device),
                token_type_ids=student_token_type_ids.to(student_device),
                attention_mask=student_attention_mask.to(student_device),
                return_hidden_and_attn=True
            )

            if use_logit_loss:
                # Sample teacher logits.
                teacher_logits = teacher_logitsbank(p_indices.to(teacher_device))
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

            # ===================== NCE_COS loss ==============================
            # Our notation follows the original paper
            # Skip embedding layers.
            student_hiddens = student_hiddens[1:]

            # get negative number.
            K = len(n_indices[0])

            # Iterate each layer.
            for s_hidden, membank, adaptive_layer in zip(
                    student_hiddens, membanks, adaptive_layers
                ):

                # Extract normalized teacher representation (anchor) features.
                # `t_hidden` : B x 768
                t_hidden = membank(p_indices.to(student_device)).T.to(student_device)

                # Get hidden batch size and dim.
                B = t_hidden.shape[0]
                D = t_hidden.shape[-1]

                # Convert student [CLS] hidden state dimension.
                s_hidden = nn.functional.normalize(
                    adaptive_layer(s_hidden[:,0,:])
                )

                #TODO: remove assertion
                assert s_hidden.shape[-1] == D, 's_hidden dim not match'

                # =============== g(z^t, z^s) =================================
                # `pos_ang_distance`: tensor of shape Bx1
                pos_ang_distance = 1 - torch.bmm(
                    s_hidden.view(B,1,D),
                    t_hidden.view(B,D,1)
                ).view(B,-1)

                # =============== \sum_{i=1}^{K}(g(n_i^t, z^s)) ===============
                indices = torch.LongTensor(n_indices).to(student_device)

                # Extract a batch of negatives
                N = []
                for idx in indices:
                    # `neg`: tensor of shape DxK
                    neg = membank(idx)
                    N.append(neg)

                # `N`: BxDxK
                N = torch.stack(N)

                # `neg_ang_distance`: tensor of shape Bx1
                # `neg_ang_distance` = \sum_{i=1}^{K}(g(n_i^t, z^s))
                neg_ang_distance = 1 - torch.bmm(s_hidden.view(B,1,D),N).view(B,-1)
                neg_ang_distance = torch.sum(neg_ang_distance,dim=-1,keepdim=True)

                #TODO: remove assertion
                assert pos_ang_distance.shape == neg_ang_distance.shape, "pos and neg ang distance shape dosen't match"

                # =========== NCE_COS loss: formula (1) ====================================
                # `batch_nce_cos_loss`: Bx1
                batch_nce_cos_loss = (2*K - (neg_ang_distance - K * pos_ang_distance)) / (2*K)
                batch_nce_cos_loss = batch_nce_cos_loss + pos_ang_distance
                # `batch_nce_cos_loss`: scalar tensor
                batch_nce_cos_loss = nce_cos_weight * torch.sum(batch_nce_cos_loss)
                # Normalize loss
                batch_nce_cos_loss = batch_nce_cos_loss / student_config.accum_step

                # Log loss.
                nce_cos_loss += batch_nce_cos_loss.item()
                loss += batch_nce_cos_loss.item()

                # Accumulate gradient.
                batch_nce_cos_loss.backward(retain_graph=True)

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch.
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
                    f'loss: {loss:.6f}' +
                    f'logits_loss: {logits_loss:.6f}' +
                    f'nce_cos_loss: {nce_cos_loss:.6f}'
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
                        '/nce_cos_loss',
                        nce_cos_loss,
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
                nce_cos_loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `student_config.ckpt_step` step.
                if step % student_config.ckpt_step == 0:
                    torch.save(
                        student_model.state_dict(),
                        os.path.join(experiment_dir, f'model-{step}.pt')
                    )

                    for i, layer in enumerate(adaptive_layers):
                        torch.save(
                            layer.state_dict(),
                            os.path.join(experiment_dir, f'adaptive-{i}-{step}.pt')
                        )

            # Start to use `logit_loss`
            if accum_step == total_contrast_step:
                use_logit_loss = True

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
    for i, layer in enumerate(adaptive_layers):
        torch.save(
            layer.state_dict(),
            os.path.join(experiment_dir, f'adaptive-{i}-{step}.pt')
        )
