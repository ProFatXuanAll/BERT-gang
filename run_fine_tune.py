"""Run fine-tune training.

Usage:
    python run_fine_tune.py ...

Run `python run_fine_tune.py -h` for help,
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

logger = logging.getLogger('fine_tune.train')
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
        help='Name of current experiment.',
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
        '--pretrained_version',
        default='',
        help='Pretrained model provided by hugginface.',
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
        '--num_class',
        default=2,
        help='Number of classes to classify.',
        required=True,
        type=int,
    )

    # Optional parameters.
    parser.add_argument(
        '--accumulation_step',
        default=1,
        help='Gradient accumulation step.',
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Training batch size.',
        type=int,
    )
    parser.add_argument(
        '--beta1',
        default=0.9,
        help="Optimizer AdamW's parameter `beta` first value.",
        type=float,
    )
    parser.add_argument(
        '--beta2',
        default=0.999,
        help="Optimizer AdamW's parameter `beta` second value.",
        type=float,
    )
    parser.add_argument(
        '--checkpoint_step',
        default=500,
        help='Checkpoint interval based on number of mini-batch.',
        type=int,
    )
    parser.add_argument(
        '--dropout',
        default=0.1,
        help='Dropout probability.',
        type=float,
    )
    parser.add_argument(
        '--epoch',
        default=0.1,
        help='Number of training epochs.',
        type=int,
    )
    parser.add_argument(
        '--eps',
        default=1e-8,
        help="Optimizer AdamW's parameter `eps`.",
        type=float,
    )
    parser.add_argument(
        '--learning_rate',
        default=3e-5,
        help="Optimizer AdamW's parameter `lr`.",
        type=float,
    )
    parser.add_argument(
        '--max_norm',
        default=1.0,
        help='Max norm of gradient.',
        type=float,
    )
    parser.add_argument(
        '--max_seq_len',
        default=512,
        help='Max sequence length for fine-tune.',
        type=int,
    )
    parser.add_argument(
        '--num_gpu',
        default=1,
        help='Number of GPUs to train.',
        type=int,
    )
    parser.add_argument(
        '--seed',
        default=42,
        help='Control random seed.',
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
        help="Optimizer AdamW's parameter `weight_decay`.",
        type=float,
    )

    args = parser.parse_args()

    config = fine_tune.config.TeacherConfig(
        accumulation_step=args.accumulation_step,
        batch_size=args.batch_size,
        beta1=args.beta1,
        beta2=args.beta2,
        checkpoint_step=args.checkpoint_step,
        dataset=args.dataset,
        dropout=args.dropout,
        epoch=args.epoch,
        eps=args.eps,
        experiment=args.experiment,
        learning_rate=args.learning_rate,
        max_norm=args.max_norm,
        max_seq_len=args.max_seq_len,
        num_class=args.num_class,
        num_gpu=args.num_gpu,
        pretrained_version=args.pretrained_version,
        seed=args.seed,
        task=args.task,
        teacher=args.teacher,
        warmup_step=args.warmup_step,
        weight_decay=args.weight_decay
    )

    logger.info(config)

    config.save()

    fine_tune.util.set_seed_by_config(
        config=config
    )

    dataset = fine_tune.util.load_dataset_by_config(
        config=config
    )

    tokenizer = fine_tune.util.load_tokenizer_by_config(
        config=config
    )

    model = fine_tune.util.load_teacher_model_by_config(
        config=config
    )

    optimizer = fine_tune.util.optimizer.load_optimizer_by_config(
        config=config,
        model=model
    )

    scheduler = fine_tune.util.scheduler.load_scheduler_by_config(
        config=config,
        dataset=dataset,
        optimizer=optimizer
    )

    fine_tune.util.train(
        config=config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer
    )
