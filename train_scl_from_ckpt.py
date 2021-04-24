r"""It is a testing script.
Use this script to find the best training schedule for SCL loss.

This script will load from a given checkpoint model which is assumed
to be trained on MSE loss before, then train this model with SCL loss.
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

def train_SCL(
    student_config: fine_tune.config.StudentConfig,
    train_dataset: fine_tune.task.Dataset,
    student_model: fine_tune.model.StudentModel,
    scl_optimizer: torch.optim.AdamW,
    student_tokenizer: transformers.PreTrainedTokenizer,
    scl_temp: float = 0.1
):
    """Train model with SCL loss.

    Parameters
    ----------
    student_config : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    train_dataset : fine_tune.task.Dataset
        Task specific dataset.
    student_model : fine_tune.model.StudentModel
        Student model.
    scl_optimizer : torch.optim.AdamW
        `torch.optim.AdamW` optimizer.
    student_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    scl_temp : float, optional
        Temperature of Supervised Contrastive Learning loss, by default 0.1
    """
    # Set model as training mode.
    student_model.train()

    # Clean all gradient.
    scl_optimizer.zero_grad()

    # Get experiment name and path for new model.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=student_config.experiment,
        model=student_config.model,
        task=student_config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Contruct custom batch sampler.
    glue_sampler = fine_tune.task.GlueBatchSampler(
        train_dataset,
        batch_size=student_config.batch_size // student_config.accum_step
    )

    # Construct data loader.
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=glue_sampler,
        collate_fn=train_dataset.create_collate_fn(),
        num_workers=os.cpu_count(),
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Create objective functions.
    scl_objective = fine_tune.contrast_util.SupConLoss(scl_temp)

    # Accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = student_config.total_step * student_config.accum_step

    # Mini-batch loss and accmulate loss.
    # Update when accumulate to `config.batch_size`.
    # For CLI and tensorboard logger.
    scl_loss = 0

    # torch.Tensor placeholder.
    accum_scl_loss = 0

    # `tqdm` CLI logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'scl_loss: {scl_loss:.6f}',
        total=student_config.total_step
    )

    # Total update times: `student_config.total_step`
    while accum_step < total_accum_step:

        # Mini-batch loop.
        for text, text_pair, label in dataloader:
            # Transform `label` to a Tensor.
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

            # Get hidden states from model.
            _, hiddens, _ = student_model(
                input_ids=input_ids.to(student_config.device),
                token_type_ids=token_type_ids.to(student_config.device),
                attention_mask=attention_mask.to(student_config.device),
                return_hidden_and_attn=True
            )

            # Get [CLS] embedding of last layer.
            last_cls = hiddens[-1][:,0,:]

            # Calculate SCL loss.
            accum_scl_loss = scl_objective(
                features=last_cls,
                labels=label.to(student_config.device)
            )

            # Normalize loss.
            accum_scl_loss = accum_scl_loss / student_config.accum_step

            # Log loss.
            scl_loss += accum_scl_loss.item()

            # Accumulate gradients.
            accum_scl_loss.backward()

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
                scl_optimizer.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'scl_loss: {scl_loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `student_config.log_step`.
                if step % student_config.log_step == 0:
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
                        '/scl_loss',
                        scl_loss,
                        step
                    )
                    writer.add_scalar(
                        f'{student_config.task}/{student_config.dataset}/{student_config.model}/lr',
                        scl_optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                scl_loss = 0

                # Clean up gradient.
                scl_optimizer.zero_grad()

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
logger = logging.getLogger('fine_tune.train_scl_from_ckpt')
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
        '--scl_temp',
        default=0.1,
        help='Temperature of Supervised Contrastive Loss.',
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
    config.weight_decay = args.weight_decay

    logger.info("New config for SCL training")
    logger.info(config)
    config.save()

    # Load optimizer.
    optimizer = fine_tune.util.load_optimizer_by_config(
        config=config,
        model=model
    )

    # Start training.
    train_SCL(
        student_config=config,
        train_dataset=dataset,
        student_model=model,
        scl_optimizer=optimizer,
        student_tokenizer=tokenizer,
        scl_temp=args.scl_temp
    )
