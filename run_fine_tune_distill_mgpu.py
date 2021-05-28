r"""Run fine-tune distillation with multi-GPU.

Usage:
    python run_fine_tune_distill_mgpu.py ...

Run `python run_fine_tune_distill_mgpu.py -h` for help, or see 'doc/fine_tune_*.md'
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
logger = logging.getLogger('fine_tune.distill')
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
        '--kd_algo',
        help='Which type of KD.\n' +
        '1. `pkd-even` for BERT-PKD with even layer mapping\n' +
        '2. `pkd-odd` for BERT-PKD with odd layer mapping\n'+
        '2. `akd` for AKD-BERT',
        required=True,
        type=str
    )
    parser.add_argument(
        '--task',
        help='Name of the distillation task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--use_classify_loss',
        help='Use logits loss during distillation',
        action='store_true'
    )
    parser.add_argument(
        '--use_hidden_loss',
        help='Distill teacher knowledge from hidden states',
        action='store_true'
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

    # Arguments of student model.
    parser.add_argument(
        '--experiment',
        help='Name of the current distillation experiment.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--model',
        help='Name of the model to distill.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--device_id',
        help='Device ID of student model.',
        required=True,
        type=int,
    )

    # Optional arguments.
    # Shared arguments.
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
        '--hard_weight',
        help='Weight of hard label.',
        default=0.8,
        type=float
    )
    parser.add_argument(
        '--soft_weight',
        help='Weight of soft label.',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--softmax_temp',
        help='Softmax temperature of soft target cross entropy loss.',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--mu',
        help='Hidden MSE loss weight',
        default=1,
        type=int
    )

    # Arguments of teacher model.
    parser.add_argument(
        '--tdevice_id',
        help='Device ID of teacher model. If not specified then load from config',
        default=-1,
        type=int
    )

    # Arguments of student model.
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
        '--d_emb',
        default=128,
        help='Embedding dimension.',
        type=int,
    )
    parser.add_argument(
        '--d_ff',
        default=3072,
        help='Transformer layers feed forward dimension.',
        type=int,
    )
    parser.add_argument(
        '--d_model',
        default=768,
        help='Transformer layers hidden dimension.',
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
        '--num_attention_heads',
        default=16,
        help='Number of attention heads in Transformer layers.',
        type=int,
    )
    parser.add_argument(
        '--num_hidden_layers',
        default=6,
        help='Number of Transformer layers.',
        type=int,
    )
    parser.add_argument(
        '--total_step',
        default=50000,
        help='Total number of step to perform training.',
        type=int,
    )
    parser.add_argument(
        '--type_vocab_size',
        default=2,
        help='BERT-like models token type embedding range.',
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

    # Construct student model configuration.
    student_config = fine_tune.config.StudentConfig(
        accum_step=args.accum_step,
        batch_size=args.batch_size,
        beta1=args.beta1,
        beta2=args.beta2,
        ckpt_step=args.ckpt_step,
        d_emb=args.d_emb,
        d_ff=args.d_ff,
        d_model=args.d_model,
        dataset=teacher_config.dataset,
        dropout=args.dropout,
        eps=args.eps,
        experiment=args.experiment,
        log_step=args.log_step,
        lr=args.lr,
        max_norm=args.max_norm,
        max_seq_len=teacher_config.max_seq_len,
        model=args.model,
        num_attention_heads=args.num_attention_heads,
        num_class=teacher_config.num_class,
        num_hidden_layers=args.num_hidden_layers,
        seed=teacher_config.seed,
        task=args.task,
        total_step=args.total_step,
        type_vocab_size=args.type_vocab_size,
        warmup_step=args.warmup_step,
        weight_decay=args.weight_decay,
        device_id=args.device_id
    )

    # Log configuration.
    logger.info(teacher_config)
    logger.info(student_config)

    # Save student config.
    student_config.save()

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

    # Load student model.
    student_model = fine_tune.util.load_student_model_by_config(
        config=student_config,
        tokenizer=student_tokenizer,
        init_from_pre_trained=True
    )

    # Load optimizer.
    optimizer = fine_tune.util.optimizer.load_optimizer_by_config(
        config=student_config,
        model=student_model
    )

    # Load scheduler.
    scheduler = fine_tune.util.scheduler.load_scheduler_by_config(
        config=student_config,
        optimizer=optimizer
    )

    # Perform disitllation.
    if args.kd_algo.lower() == 'pkd-even':
        logger.info("Train BERT-PKD with even layer mapping")
        fine_tune.util.train_PKD(
            teacher_config=teacher_config,
            student_config=student_config,
            dataset=dataset,
            teacher_model=teacher_model,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            soft_weight=args.soft_weight,
            mse_weight=args.mu,
            softmax_temp=args.softmax_temp,
            layer_mapping='even'
        )
    elif args.kd_algo.lower() == 'pkd-odd':
        logger.info("Train BERT-PKD with odd layer mapping")
        fine_tune.util.train_PKD(
            teacher_config=teacher_config,
            student_config=student_config,
            dataset=dataset,
            teacher_model=teacher_model,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            soft_weight=args.soft_weight,
            mse_weight=args.mu,
            softmax_temp=args.softmax_temp,
            layer_mapping='odd'
        )
    elif args.kd_algo.lower() == 'akd':
        # perform distillation.
        logger.info("Train AKD-BERT")
        fine_tune.util.distill_mgpu(
            teacher_config=teacher_config,
            student_config=student_config,
            dataset=dataset,
            teacher_model=teacher_model,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            use_classify_loss=args.use_classify_loss,
            use_hidden_loss=args.use_hidden_loss,
            alpha=args.soft_weight,
            gamma=args.hard_weight,
            mu=args.mu,
            softmax_temp=args.softmax_temp
        )
    else:
        raise ValueError(f"Un supported kd algo: {args.kd_algo}")
