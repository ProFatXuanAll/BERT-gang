r"""Run fine-tune model soft-target generation.

Usage:
    python run_fine_tune_gen_soft_target.py ...

Run `python run_fine_tune_gen_soft_target.py -h` for help,
or see doc/fine_tune_*.md for more information.
"""

# built-in modules

import argparse
import logging

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.gen_soft_target')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)

# Filter out message not begin with name 'fine_tune'.
for handler in logging.getLogger().handlers:
    handler.addFilter(logging.Filter('fine_tune'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        '--experiment',
        default='',
        help='Name of the experiment to evalutate.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--teacher',
        default='',
        help="Teacher model's name.",
        required=True,
        type=str,
    )
    parser.add_argument(
        '--task',
        default='',
        help='Name of fine-tune task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        default='',
        help='Dataset name of particular fine-tune task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--ckpt',
        default=0,
        help='Which checkpoint to generate soft target.',
        required=True,
        type=int,
    )

    # Optional parameters.
    parser.add_argument(
        '--batch_size',
        default=0,
        help='Soft target batch size.',
        type=int,
    )

    args = parser.parse_args()

    config = fine_tune.config.TeacherConfig.load(
        experiment=args.experiment,
        task=args.task,
        teacher=args.teacher
    )

    # Change batch size for faster evaluation.
    if args.batch_size:
        config.batch_size = args.batch_size

    logger.info(config)

    fine_tune.util.set_seed_by_config(
        config=config
    )

    dataset = fine_tune.util.load_dataset(
        dataset=args.dataset,
        task=args.task,
    )

    fine_tune.util.gen_soft_target(
        ckpt=args.ckpt,
        config=config,
        dataset=dataset
    )
