r"""Run fine-tune model logits generation.

Usage:
    python run_fine_tune_gen_logits.py ...

Run `python run_fine_tune_gen_logits.py -h` for help, or see
'doc/fine_tune_*.md' for more information.
"""

# built-in modules

import argparse
import logging
import os

# 3rd-party modules

import torch

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.gen_logits')
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
        help='Name of the previous experiment to generate logits.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--model',
        help='Name of the model to generate logits.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--task',
        help='Name of fine-tune task to generate logits on.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name of the fine-tune task to generate logits on.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--ckpt',
        help='Model checkpoint to generate logits.',
        required=True,
        type=int,
    )

    # Optional parameters.
    parser.add_argument(
        '--batch_size',
        default=0,
        help='Generate logits batch size.',
        type=int,
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

    # Change batch size for faster generation.
    if args.batch_size:
        config.batch_size = args.batch_size

    # Set dataset to generate logits on.
    config.dataset = args.dataset

    # Log configuration.
    logger.info(config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Load dataset for generating logits.
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

    # Get experiment name and model name.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    model_name = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name,
        f'model-{args.ckpt}.pt'
    )

    # Load model from checkpoint.
    model.load_state_dict(torch.load(model_name))

    # Generate logits.
    fine_tune.util.gen_logits(
        config=config,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer
    )
