r"""Run some probing experiments on LAD.
For example what will happend if we only use last gate networks output as
training objective.
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
logger = logging.getLogger('fine_tune.probing_lad')
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
        '--task',
        help='Name of the distillation task.',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--probing_exp',
        help='What type of probing experiment to be conducted.',
        required=True,
        type=str,
    )

    # Arguments of teacher model.
    parser.add_argument(
        '--gate_indices',
        help='Which gate layers a student model should learn.' +
        'For example type in `2 4 6 8 10` mean those gate layers '+
        'will be learned by student model.',
        required=True,
        type=str
    )

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
        '--student_indices',
        help='Which student layers will be updated.'+
        'For example type in `1 2 3 4 5` means params '+
        'of those student layers will be updated.',
        required=True,
        type=str
    )
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

    # Arguments of gate networks.
    parser.add_argument(
        '--gate_device_id',
        help='Device ID of gate networks.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--gate_beta1',
        default=0.9,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float
    )
    parser.add_argument(
        '--gate_beta2',
        default=0.999,
        help="Optimizer `torch.optim.AdamW`'s beta coefficients.",
        type=float,
    )
    parser.add_argument(
        '--gate_eps',
        default=1e-8,
        help="Optimizer `torch.optim.AdamW`'s epsilon.",
        type=float,
    )
    parser.add_argument(
        '--gate_lr',
        default=1e-5,
        help="Optimizer `torch.optim.AdamW`'s learning rate.",
        type=float,
    )
    parser.add_argument(
        '--gate_max_norm',
        default=1.0,
        help='Maximum norm of gradient.',
        type=float,
    )
    parser.add_argument(
        '--gate_total_step',
        required=True,
        help='Total number of step to train gate.',
        type=int,
    )
    parser.add_argument(
        '--gate_warmup_step',
        required=True,
        help='Linear scheduler warmup step of gate network scheduler.',
        type=int,
    )
    parser.add_argument(
        '--gate_weight_decay',
        default=0.01,
        help="Optimizer `torch.optim.AdamW` weight decay regularization.",
        type=float,
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
        type=float
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
        '--seed',
        default=42,
        help="Random seed of student model.",
        type=int
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

    # Parse student and teacher indices.
    gate_indices = [int(index)-1 for index in args.gate_indices.split(',')]
    student_indices = [int(index)-1 for index in args.student_indices.split(',')]
    logger.info("Teacher indices: %s", gate_indices)
    logger.info("Student indices: %s", student_indices)

    # Load fine-tune teacher model configuration.
    teacher_config = fine_tune.config.TeacherConfig.load(
        experiment=args.teacher_exp,
        model=args.tmodel,
        task=args.task
    )

    # Sync batch size and accumulation steps.
    teacher_config.seed = args.seed
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
        seed=args.seed,
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
        config=student_config
    )

    # Load distillation dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=teacher_config
    )

    logger.info("Load teacher and student tokenizer")
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
    logger.info("Load teacher model from given checkpoint: %s", model_name)
    teacher_model.load_state_dict(torch.load(model_name, map_location=teacher_config.device))

    # Load student model.
    logger.info("Load student model")
    student_model = fine_tune.util.load_student_model_by_config(
        config=student_config,
        tokenizer=student_tokenizer
    )

    logger.info("Load student model's optimizer and scheduler.")
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

    # Init gate network block config.
    gate_config = fine_tune.config.GateConfig(
        dimension=student_config.d_model,
        max_seq_length=student_config.max_seq_len,
        beta1=args.gate_beta1,
        beta2=args.gate_beta2,
        eps=args.gate_eps,
        total_step=args.gate_total_step,
        warmup_step=args.gate_warmup_step,
        lr=args.gate_lr,
        max_norm=args.gate_max_norm,
        weight_decay=args.gate_weight_decay,
        device_id=args.gate_device_id
    )

    # Log configuration
    logger.info(gate_config)

    # Save config.
    gate_config.save(
        os.path.join(
            fine_tune.path.FINE_TUNE_EXPERIMENT,
            f'{args.experiment}_{args.model}_{args.task}'
        )
    )

    if args.probing_exp.lower() == 'lad_user_defined':
        # Load gate networks.
        logger.info("Load gate networks by teacher and student config.")
        gate_networks = fine_tune.util.load_gate_networks_by_config(
            teacher_config=teacher_config,
            gate_config=gate_config
        )

        logger.info("Load gate networks' optimizer and scheduler")
        # Load optimizer.
        gates_optimizer = fine_tune.util.load_gate_networks_optimizer(
            betas=gate_config.betas,
            eps=gate_config.eps,
            lr=gate_config.lr,
            weight_decay=gate_config.weight_decay,
            gate_networks=gate_networks
        )

        # Load scheduler
        gates_scheduler = fine_tune.util.load_gate_networks_scheduler(
            optimizer=gates_optimizer,
            total_step=gate_config.total_step,
            warmup_step=gate_config.warmup_step
        )

        logger.info("Run probing experiments: `lad_user_defined`")
        fine_tune.util.train_lad_user_defined(
            teacher_config=teacher_config,
            student_config=student_config,
            gate_config=gate_config,
            dataset=dataset,
            teacher_model=teacher_model,
            student_model=student_model,
            gate_networks=gate_networks,
            optimizer=optimizer,
            scheduler=scheduler,
            gates_optimizer=gates_optimizer,
            gates_scheduler=gates_scheduler,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            gate_indices=gate_indices,
            student_indices=student_indices,
            alpha=args.soft_weight,
            gamma=args.hard_weight,
            mu=args.mu,
            softmax_temp=args.softmax_temp,
        )
    elif args.probing_exp.lower() == 'partial_lad':
        # Load gate networks.
        logger.info("Load gate networks.")
        gate_networks = fine_tune.util.load_gate_networks(
            num_layers=max(gate_indices)+1,
            dimension=gate_config.dimension,
            seq_length=gate_config.max_seq_length,
            device=gate_config.device
        )

        logger.info("Load gate networks' optimizer and scheduler")
        # Load optimizer.
        gates_optimizer = fine_tune.util.load_gate_networks_optimizer(
            betas=gate_config.betas,
            eps=gate_config.eps,
            lr=gate_config.lr,
            weight_decay=gate_config.weight_decay,
            gate_networks=gate_networks
        )

        # Load scheduler
        gates_scheduler = fine_tune.util.load_gate_networks_scheduler(
            optimizer=gates_optimizer,
            total_step=gate_config.total_step,
            warmup_step=gate_config.warmup_step
        )

        logger.info("Run probing experiments: `partial_lad`")
        fine_tune.util.train_partial_lad(
            teacher_config=teacher_config,
            student_config=student_config,
            gate_config=gate_config,
            dataset=dataset,
            teacher_model=teacher_model,
            student_model=student_model,
            gate_networks=gate_networks,
            optimizer=optimizer,
            scheduler=scheduler,
            gates_optimizer=gates_optimizer,
            gates_scheduler=gates_scheduler,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            gate_indices=gate_indices,
            student_indices=student_indices,
            alpha=args.soft_weight,
            gamma=args.hard_weight,
            mu=args.mu,
            softmax_temp=args.softmax_temp,
        )
    else:
        raise ValueError(f"Unsupported probing experiments: {args.probing_exp}")
