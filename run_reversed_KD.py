r"""Run reversed knowledge distillation (student teach teacher)

Usage:
    python run_reversed_KD.py ...

Run `python run_reversed_KD.py -h` for help, or see 'doc/fine_tune_*.md'
for more information.
"""

# built-in modules

import os
import argparse
import logging

# 3rd party modules

import torch

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.reversed_distill')
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

    # Shared arguments.
    parser.add_argument(
        '--experiment',
        help='Name of the current fine-tune experiment.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--task',
        help='Name of the distillation task.',
        required=True,
        type=str,
    )

    # Arguments of teacher model.
    parser.add_argument(
        '--teacher_exp',
        help='Experiment name of the fine-tuned teacher model',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tmodel',
        help='Name of the teacher model',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tckpt',
        help='Checkpoint of teacher model',
        required=True,
        type=int,
    )

    # Arguments of student model.
    parser.add_argument(
        '--student_exp',
        help='Experiment name of the distilled student model',
        required=True,
        type=str
    )
    parser.add_argument(
        '--smodel',
        help='Name of the student model',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--sckpt',
        help='Checkpoint of student model',
        required=True,
        type=int
    )

    # Optional arguments.

    # Teacher arguments.
    parser.add_argument(
        '--teacher_device',
        default=-1,
        help='Device ID of teacher model.',
        type=int
    )

    # Student arguments.
    parser.add_argument(
        '--student_device',
        default=-1,
        help='Device ID of student model.',
        type=int
    )

    # Shared arguments.
    # We set defaul value of follwing args to '0'
    # It means we will direct load their value from teacher config
    parser.add_argument(
        '--accum_step',
        default=0,
        help='Gradient accumulation step.',
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=0,
        help='Training batch size.',
        type=int,
    )
    parser.add_argument(
        '--ckpt_step',
        default=0,
        help='Checkpoint save interval.',
        type=int,
    )
    parser.add_argument(
        '--log_step',
        default=0,
        help='Logging interval.',
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=0.0,
        help="Optimizer `torch.optim.AdamW`'s learning rate.",
        type=float,
    )
    parser.add_argument(
        '--total_step',
        default=0,
        help='Total number of step to perform training.',
        type=int,
    )
    parser.add_argument(
        '--warmup_step',
        default=0,
        help='Linear scheduler warmup step.',
        type=int,
    )
    parser.add_argument(
        '--softmax_temp',
        default=1.0,
        help='Softmax temperature.',
        type=float
    )
    parser.add_argument(
        '--soft_target_weight',
        default=0.2,
        help='Weihgt of soft target cross entropy loss',
        type=float
    )

    # Parse arguments.
    args = parser.parse_args()

    # Load teacher model configuration.
    teacher_config = fine_tune.config.TeacherConfig.load(
        experiment=args.teacher_exp,
        model=args.tmodel,
        task=args.task
    )

    # Load student model configuration.
    student_config = fine_tune.config.StudentConfig.load(
        experiment=args.student_exp,
        model=args.smodel,
        task=args.task
    )

    # Set new device_id if needed.
    if args.student_device > -1:
        student_config.device_id = args.student_device
    if args.teacher_device > -1:
        teacher_config.device_id = args.teacher_device

    # Log configuration.
    logger.info(teacher_config)
    logger.info(student_config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=teacher_config
    )

    # Load distillation dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=teacher_config
    )

    # Load teacher and student tokenizer.
    teacher_tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
        config=teacher_config
    )
    student_tokenizer = fine_tune.util.load_student_tokenizer_by_config(
        config=student_config
    )

    # Load teacher model from given checkpoint.
    teacher_model = fine_tune.util.load_teacher_model_by_config(
        config=teacher_config
    )
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=teacher_config.experiment,
        model=teacher_config.model,
        task=teacher_config.task
    )
    model_name = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name,
        f'model-{args.tckpt}.pt'
    )

    # Load model from checkpoint.
    teacher_model.load_state_dict(torch.load(model_name, map_location=teacher_config.device))


    # Load student model from given checkpoint.
    student_model = fine_tune.util.load_student_model_by_config(
        config=student_config,
        tokenizer=student_tokenizer
    )
    s_experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=student_config.experiment,
        model=student_config.model,
        task=student_config.task
    )
    s_model_name = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        s_experiment_name,
        f'model-{args.sckpt}.pt'
    )

    # Load model from checkpoint.
    student_model.load_state_dict(torch.load(s_model_name, map_location=student_config.device))

    # Sync training hyperparameters.
    if args.accum_step > 0:
        teacher_config.accum_step = args.accum_step
        student_config.accum_step = args.accum_step
    if args.batch_size > 0:
        teacher_config.batch_size = args.batch_size
        student_config.batch_size = args.batch_size
    if args.ckpt_step > 0:
        teacher_config.ckpt_step = args.ckpt_step
    if args.log_step > 0:
        teacher_config.log_step = args.log_step
    if args.lr > 0:
        teacher_config.lr = args.lr
    if args.total_step > 0:
        teacher_config.total_step = args.total_step
    if args.warmup_step > 0:
        teacher_config.warmup_step = args.warmup_step

    # Change experiment name of Teacher config.
    # Cause we need to save new teacher model to differnet path.
    teacher_config.experiment = args.experiment
    logger.info("Change experiment name of Teacher config.")
    logger.info("New Teacher config")
    logger.info(teacher_config)
    teacher_config.save()

    # Load optimizer.
    optimizer = fine_tune.util.optimizer.load_optimizer_by_config(
        config=teacher_config,
        model=teacher_model
    )

    # Load scheduler.
    scheduler = fine_tune.util.scheduler.load_scheduler_by_config(
        config=teacher_config,
        optimizer=optimizer
    )

    # Peform Reversed-KD
    logger.info("Perform reversed knowledge distillation")
    fine_tune.util.reversed_KD(
        teacher_config=teacher_config,
        student_config=student_config,
        dataset=dataset,
        teacher_model=teacher_model,
        student_model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        alpha=args.soft_target_weight,
        softmax_temp=args.softmax_temp
    )
