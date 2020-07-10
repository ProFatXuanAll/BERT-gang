"""Run fine-tune evaluation.

Usage:
    python run_fine_tune_eval.py ...

Run `python run_fine_tune_eval.py -h` for help,
or see doc/fine_tune_*.md for more information.
"""

# built-in modules

import argparse
import logging

# my own modules

import fine_tune

# Filter out message not begin with name 'fine_tune'.
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(logging.Filter('fine_tune'))

logger = logging.getLogger('fine_tune.eval')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)


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

    # Optional parameters.
    parser.add_argument(
        '--batch_size',
        default=0,
        help='Evaluation batch size.',
        type=int,
    )

    args = parser.parse_args()

    config = fine_tune.config.TeacherConfig.load(
        experiment=args.experiment,
        task=args.task,
        teacher=args.teacher
    )

    # Change batch size for faster evaluation.
    if not args.batch_size:
        config.batch_size = args.batch_size

    logger.info(config)

    fine_tune.util.set_seed_by_config(
        config=config
    )

    dataset = fine_tune.util.load_dataset(
        dataset=args.dataset,
        task=args.task,
    )

    max_acc, max_acc_ckpt = fine_tune.util.evaluation(
        config=config,
        dataset=dataset
    )

    logger.info('max accuracy:            %f', max_acc)
    logger.info('max accuracy checkpoint: %s', max_acc_ckpt)
