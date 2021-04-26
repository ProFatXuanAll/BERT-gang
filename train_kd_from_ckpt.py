r"""It is a temporary script.
Use this script to train linear classifier on top of BERT student model.

This script will load from a given checkpoint model which is assumed
to be trained on MSE loss and SCE loss before,
then train this model with Cross Entropy loss.
"""

# built-in modules

import os
import argparse
import logging

# 3rd party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import transformers

from tqdm import tqdm
# my own modules

import fine_tune

def train_kd(
    t_config: fine_tune.config.TeacherConfig,
    student_config: fine_tune.config.StudentConfig,
    train_dataset: fine_tune.task.Dataset,
    t_model: fine_tune.model.TeacherModel,
    student_model: fine_tune.model.StudentModel,
    kd_optimizer: torch.optim.AdamW,
    kd_scheduler: torch.optim.lr_scheduler.LambdaLR,
    teacher_tokenizer: transformers.PreTrainedTokenizer,
    student_tokenizer: transformers.PreTrainedTokenizer,
    alpha: float = 0.5,
    softmax_temp: float = 1.0
):
    """Train knowledge distillation loss from given fine-tuned teacher and
    student model which has been trained on MSE and SCL loss.
    This function may use two gpu-device.

    Notes
    ----------
    .. math::
        L_{KD} = (1-\alpha) * L_{CE}(y, y^s) + \alpha * L_{KL_Div}(y^t, t^s)

    Parameters
    ----------
    t_config : fine_tune.config.TeacherConfig
        `fine_tune.config.TeacherConfig` class which attributes are used
        for experiment setup.
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    train_dataset : fine_tune.task.Dataset
        Task specific dataset.
    t_model : fine_tune.model.TeacherModel
        A fine-tuned teacher model which is used to generate soft targets.
    student_model : fine_tune.model.StudentModel
        Model which will perform disitllation according to
        outputs from given teacher model.
    kd_optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    kd_scheduler : torch.optim.lr_scheduler.LambdaLR
        Linear warmup scheduler provided by `transformers` package.
    teacher_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `teacher_model`.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    alpha : float, optional
        Weight of soft target loss, by default 0.5
    softmax_temp : float, optional
        Softmax temperature, by default 1.0
    """

    # Set teacher model as evaluation mode.
    t_model.eval()

    # Set student model as training mode.
    student_model.train()

    # Clean all gradient.
    kd_optimizer.zero_grad()

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
        train_dataset,
        batch_size=student_config.batch_size // student_config.accum_step,
        collate_fn=train_dataset.create_collate_fn(),
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
    criterion = fine_tune.objective.distill_loss

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    logits_loss = 0

    # torch.Tensor placeholder.
    accum_logits_loss=0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'logits_loss: {logits_loss:.6f}',
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
                max_length=t_config.max_seq_len,
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

            # Get soft targets.
            with torch.no_grad():
                teacher_logits, _ = t_model(
                    input_ids = teacher_input_ids.to(t_config.device),
                    token_type_ids=teacher_token_type_ids.to(t_config.device),
                    attention_mask=teacher_attention_mask.to(t_config.device)
                )

            # Get ouput logits from student models.
            student_logits, _ = student_model(
                input_ids = student_input_ids.to(student_config.device),
                token_type_ids=student_token_type_ids.to(student_config.device),
                attention_mask=student_attention_mask.to(student_config.device)
            )

            accum_logits_loss = criterion(
                hard_target=label.to(student_config.device),
                teacher_logits=teacher_logits.to(student_config.device),
                student_logits=student_logits,
                alpha=alpha,
                softmax_temp=softmax_temp
            )
            # Normalize loss.
            accum_logits_loss = accum_logits_loss / student_config.accum_step

            # Log loss.
            logits_loss += accum_logits_loss.item()

            # Accumulate gradients.
            accum_logits_loss.backward()

            # Increment accumulation step.
            accum_step += 1

            # Peform gradient descend when achieve actual mini-batch.
            if accum_step % student_config.accum_step == 0:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    student_config.max_norm
                )

                # Gradient descend.
                kd_optimizer.step()

                # Update learning rate.
                kd_scheduler.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'logits_loss: {logits_loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `student_config.log_step`.
                if step % student_config.log_step == 0:
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
                        '/logits_loss',
                        logits_loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}/lr',
                        kd_optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                logits_loss = 0

                # Clean up gradient.
                kd_optimizer.zero_grad()

                # Save model for each `student_config.ckpt_step`.
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





# Get main logger.
logger = logging.getLogger('fine_tune.train_ce_from_ckpt')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)

# Filter out message not begin with name 'fine_tune'.
for handler in logging.getLogger().handlers:
    handler.addFilter(logging.Filter('fine_tune'))

if __name__ == "__main__":
    # Parse arguments from STDIN.
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        '--teacher_exp',
        help='Experiment name of the fine-tuned teacher model',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tmodel',
        help='Name of the teacher model to transfer knowledge',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tckpt',
        help='Checkpoint of teacher model to generate logits, hidden states and attentions',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--experiment',
        help='Name of the current fine-tune experiment.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--src_experiment',
        help='Name of the source experiment name.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--src_ckpt',
        help='Checkpoint of source model.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--model',
        help='Name of the model to be trained',
        required=True,
        type=str
    )
    parser.add_argument(
        '--task',
        help='Name of the fine-tune task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--device_id',
        help='Device ID of student model.',
        required=True,
        type=int
    )

    # Optional arguments.
    parser.add_argument(
        '--tdevice_id',
        help='Device ID of teacher model. If not specified then load from config',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--softmax_temp',
        default=1.0,
        help='Temperature of softmax function.',
        type=float
    )
    parser.add_argument(
        '--soft_weight',
        default=0.5,
        help='Weight of soft target loss.',
        type=float
    )
    parser.add_argument(
        '--accum_step',
        default=1,
        help='Gradient accumulation step.',
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Distillation batch size.',
        type=int,
    )
    parser.add_argument(
        '--beta1',
        default=0.9,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float,
    )
    parser.add_argument(
        '--beta2',
        default=0.999,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float,
    )
    parser.add_argument(
        '--ckpt_step',
        default=1000,
        help='Checkpoint save interval.',
        type=int,
    )
    parser.add_argument(
        '--dropout',
        default=0.1,
        help='Dropout probability.',
        type=float,
    )
    parser.add_argument(
        '--eps',
        default=1e-8,
        help="Optimizer `torch.optim.AdamW`'s epsilon.",
        type=float,
    )
    parser.add_argument(
        '--log_step',
        default=500,
        help='Logging interval.',
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=3e-5,
        help="Optimizer `torch.optim.AdamW`'s learning rate.",
        type=float,
    )
    parser.add_argument(
        '--max_norm',
        default=1.0,
        help='Maximum norm of gradient.',
        type=float,
    )
    parser.add_argument(
        '--total_step',
        default=50000,
        help='Total number of step to perform training.',
        type=int,
    )
    parser.add_argument(
        '--warmup_step',
        default=10000,
        help='Linear scheduler warmup step.',
        type=int,
    )
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        help="Optimizer `torch.optim.AdamW` weight decay regularization.",
        type=float,
    )

    # Parse arguments.
    args = parser.parse_args()

    # Load source model configuration.
    config = fine_tune.config.StudentConfig.load(
        experiment=args.src_experiment,
        model=args.model,
        task=args.task
    )

    # Log source configuration.
    logger.info("Source model configuration:")
    logger.info(config)

    # Load fine-tune teacher model configuration.
    teacher_config = fine_tune.config.TeacherConfig.load(
        experiment=args.teacher_exp,
        model=args.tmodel,
        task=args.task
    )

    # Sync batch size and accumulation steps.
    teacher_config.batch_size = args.batch_size
    teacher_config.accum_step = args.accum_step

    # Set new device ID for teacher model if needed.
    if args.tdevice_id > -1:
        teacher_config.device_id = args.tdevice_id

    # Log teacher configuration.
    logger.info("Teacher model configuration:")
    logger.info(teacher_config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Set new device id.
    config.device_id = args.device_id

    # Load training dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=config
    )

    # Load teacher tokenizer and model.
    teacher_tknr = fine_tune.util.load_teacher_tokenizer_by_config(
        config=teacher_config
    )
    teacher_model = fine_tune.util.load_teacher_model_by_config(
        config=teacher_config
    )

    # Load teacher model from given checkpoint.
    texp_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=teacher_config.experiment,
        model=teacher_config.model,
        task=teacher_config.task
    )
    tmodel_name = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        texp_name,
        f'model-{args.tckpt}.pt'
    )

    logger.info("Load teacher checkpoint from %s", tmodel_name)

    teacher_model.load_state_dict(
        torch.load(
            tmodel_name,
            map_location=teacher_config.device
        )
    )

    # Load tokenizer and model.
    tokenizer = fine_tune.util.load_student_tokenizer_by_config(
        config=config
    )
    model = fine_tune.util.load_student_model_by_config(
        config=config,
        tokenizer=tokenizer
    )

    # Load parameters from checkpoint.
    src_exp_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    src_exp_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        src_exp_name
    )
    src_model_fname = os.path.join(src_exp_dir, f'model-{args.src_ckpt}.pt')

    logger.info("Load checkpoint from %s", src_model_fname)

    model.load_state_dict(
        torch.load(
            src_model_fname,
            map_location=config.device
        )
    )

    # Set new config for SCL training and save it.
    config.experiment = args.experiment
    config.accum_step = args.accum_step
    config.batch_size = args.batch_size
    config.beta1 = args.beta1
    config.beta2 = args.beta2
    config.ckpt_step = args.ckpt_step
    config.dropout = args.dropout
    config.eps = args.eps
    config.log_step = args.log_step
    config.lr = args.lr
    config.max_norm = args.max_norm
    config.total_step = args.total_step
    config.warmup_step = args.warmup_step
    config.weight_decay = args.weight_decay

    logger.info("New config for KD training")
    logger.info(config)
    config.save()

    # Load optimizer.
    optimizer = fine_tune.util.load_optimizer_by_config(
        config=config,
        model=model
    )

    # Load scheduler.
    scheduler = fine_tune.util.load_scheduler_by_config(
        config=config,
        optimizer=optimizer
    )

    # Start training.
    train_kd(
        t_config=teacher_config,
        student_config=config,
        train_dataset=dataset,
        t_model=teacher_model,
        student_model=model,
        kd_optimizer=optimizer,
        kd_scheduler=scheduler,
        teacher_tokenizer=teacher_tknr,
        student_tokenizer=tokenizer,
        alpha=args.soft_weight,
        softmax_temp=args.softmax_temp
    )
