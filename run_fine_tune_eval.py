r"""Run fine-tune evaluation.

Usage:
    python run_fine_tune_eval.py ...

Run `python run_fine_tune_eval.py -h` for help, or see 'doc/fine_tune_*.md' for
more information.
"""

# built-in modules

import argparse
import logging
import os
import re

# 3rd-party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.eval')
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)

# Filter out message not begin with name 'fine_tune'.
for handler in logging.getLogger().handlers:
    handler.addFilter(logging.Filter('fine_tune'))

if __name__ == '__main__':
    # Parse arguments from STDIN.
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        '--experiment',
        help='Name of the previous experiment to evalutate.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--model',
        help='Name of the model to evaluate.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--task',
        help='Name of the fine-tune task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name of the fine-tune task.',
        required=True,
        type=str,
    )

    # Optional parameters.
    parser.add_argument(
        '--batch_size',
        default=0,
        help='Evaluation batch size.',
        type=int,
    )
    parser.add_argument(
        '--device_id',
        default=-1,
        help='Run evaluation on dedicated device.',
        type=int,
    )
    parser.add_argument(
        '--ckpt',
        default=0,
        help='Start evaluation from specified checkpoint',
        type=int
    )

    # Parse arguments.
    args = parser.parse_args()

    # Load fine-tune teacher model configuration.
    # `fine_tune.config.TeacherConfig.load` will trigger `TypeError` if the
    # actual configuration file is saved by `fine_tune.config.StudentConfig`.
    try:
        config = fine_tune.config.TeacherConfig.load(
            experiment=args.experiment,
            model=args.model,
            task=args.task
        )
    # Load fine-tune distillation student model configuration.
    except TypeError:
        config = fine_tune.config.StudentConfig.load(
            experiment=args.experiment,
            model=args.model,
            task=args.task
        )

    # Change batch size for faster evaluation.
    if args.batch_size:
        config.batch_size = args.batch_size

    # Check user specify device or not.
    if args.device_id > -1:
        config.device_id = args.device_id
    logger.info("Use device: %s to run evaluation", config.device_id)

    # Set evaluation dataset.
    config.dataset = args.dataset

    # Log configuration.
    logger.info(config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Load fine-tune / distillation dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=config
    )

    # Load teacher tokenizer and model.
    if isinstance(config, fine_tune.config.TeacherConfig):
        tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
            config=config
        )
        model = fine_tune.util.load_teacher_model_by_config(
            config=config
        )
    # Load student tokenizer and model.
    else:
        tokenizer = fine_tune.util.load_student_tokenizer_by_config(
            config=config
        )
        model = fine_tune.util.load_student_model_by_config(
            config=config,
            tokenizer=tokenizer
        )

    # Get experiment name and path.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Get all checkpoint file names.
    ckpt_pattern = r'model-(\d+)\.pt'
    all_ckpts = sorted(map(
        lambda file_name: int(re.match(ckpt_pattern, file_name).group(1)),
        filter(
            lambda file_name: re.match(ckpt_pattern, file_name),
            os.listdir(experiment_dir)
        ),
    ))

    # Filt unnecessary checkpoint.
    all_ckpts = list(filter(lambda ckpt: ckpt >= args.ckpt, all_ckpts))

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Record maximum accuracy and its respective checkpoint.
    max_acc = 0.0
    max_acc_ckpt = 0

    # Evaluate every checkpoints.
    for ckpt in all_ckpts:

        # Clean all gradient.
        model.zero_grad()

        # Load model from checkpoint.
        model.load_state_dict(
            torch.load(
                os.path.join(experiment_dir,f'model-{ckpt}.pt'),
                map_location=config.device
            )
        )

        # Calculate accuracy.
        if config.amp:
            acc = fine_tune.util.amp_evaluation(
                config=config,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer
            )
        else:
            acc = fine_tune.util.evaluation(
                config=config,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer
            )

        # Update max accuracy.
        if max_acc <= acc:
            max_acc = acc
            max_acc_ckpt = ckpt

        # Log accuracy.
        writer.add_scalar(
            f'{config.task}/{config.dataset}/accuracy',
            acc,
            ckpt
        )

    # Release IO resources.
    writer.flush()
    writer.close()

    # Log maximum accuracy.
    logger.info('max accuracy:            %f', max_acc)
    logger.info('max accuracy checkpoint: %d', max_acc_ckpt)
