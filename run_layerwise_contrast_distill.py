r"""Run fine-tune two stage contrastive distillation layer wise.

Usage:
    python run_layerwise_contrast_distill.py

Run `python run_layerwise_contrast.py -h` for help, or see documentation
for more information.
"""

# built-in modules

import os
import argparse
import logging
import re

# 3rd party modules

import torch
from tqdm import tqdm

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.contrast_distill_layerwise')
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

    # Required shared arguments.
    parser.add_argument(
        '--task',
        help='Name of the distillation task.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--neg_num',
        help='Number of negative samples.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--contrast_steps',
        help='Training iterations for contrastive loss only',
        required=True,
        type=int
    )

    # Required arguments of teacher model.
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

    # Required arguments of student model.
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

    # Optional shared arguments.
    parser.add_argument(
        '--embedding_type',
        default='cls',
        help='`cls`: store teacher cls embedding to memory bank \n'+
        '`mean`: store average BERT output embedding to memory bank',
        type=str
    )
    parser.add_argument(
        '--softmax_temp',
        help='Softmax temeprature',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--contrast_temp',
        help='Temperature term of InfoNCE loss',
        default=0.07,
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
        '--contrast_weight',
        default=1,
        help='Weight of contrast loss.',
        type=float
    )
    parser.add_argument(
        '--defined_by_label',
        help='Use label information to define negative samples.',
        action='store_true'
    )

    # Optional arguments of teacher model.
    parser.add_argument(
        '--teacher_device',
        help='Teacher model device id',
        default=-1,
        type=int
    )

    # Optional arguments of student model.
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
        '--soft_label_weight',
        default=0.2,
        help="loss weight (alpha) of soft target cross entropy." +
            "See 'distill_loss' in `fine_tune.objective`",
        type=float
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

    # Check teacher model device.
    if args.teacher_device > -1:
        teacher_config.device_id = args.teacher_device
        logger.info("Move teacher logits to device %s", teacher_config.device)

    # Construct student model configuration.
    student_config = fine_tune.config.StudentConfig(
        accum_step=args.accum_step,
        amp=False,
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

    # Load contrast distillation dataset.
    dataset = fine_tune.util.load_contrast_dataset_by_config(
        config=teacher_config,
        neg_num=args.neg_num,
        defined_by_label=args.defined_by_label
    )

    # Load teacher and student tokenizer.
    # teacher_tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
    #     config=teacher_config
    # )
    student_tokenizer = fine_tune.util.load_student_tokenizer_by_config(
        config=student_config
    )

    # Load teacher output logits bank from given checkpoint.
    logger.info("Load teacher logits.")

    teacher_logits = fine_tune.model.Logitsbank(
        N = len(dataset),
        C = teacher_config.num_class
    )

    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=teacher_config.experiment,
        model=teacher_config.model,
        task=teacher_config.task
    )
    logits_fname = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name,
        f'logitsbank-{args.tckpt}.pt'
    )

    teacher_logits.load_state_dict(torch.load(logits_fname))
    teacher_logits.to(teacher_config.device)

    logger.info("Loading teacher logits complete.")

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

    logger.info("Init memory banks")

    # Build memory bank.
    if 'bert-base' in teacher_config.ptrain_ver:
        dim = 768
    elif 'bert-large' in teacher_config.ptrain_ver:
        dim = 1024

    membanks = [
        fine_tune.contrast_util.Memorybank(
            N=len(dataset),
            dim=dim,
            embd_type=args.embedding_type
        ) for _ in range(args.num_hidden_layers)
    ]

    # Load from file.
    t_membank_path = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    logger.info("Extract membank files")

    # Filter out membank file number and sort it.
    if args.embedding_type.lower() == 'cls':
        file_pattern = r'membank(\d+)\_cls.pt'
    else:
        file_pattern = r'membank(\d+)\_mean.pt'
    all_mem_files = sorted(map(
        lambda file_name: int(re.match(file_pattern, file_name).group(1)),
        filter(
            lambda file_name: re.match(file_pattern, file_name),
            os.listdir(t_membank_path)
        )
    ))

    step = len(all_mem_files) // args.num_hidden_layers

    all_mem_files = all_mem_files[step-1::step]

    logger.info("Load memory banks from .pt files")
    try:
        for l, (i, membank) in enumerate(tqdm(zip(all_mem_files, membanks), total=len(all_mem_files))):
            membank.load_state_dict(torch.load(os.path.join(
                t_membank_path,
                f'membank{i}_{args.embedding_type.lower()}.pt'
            )))
            #TODO: refactor
            # if l < 6:
            #     membank.to(fine_tune.util.genDevice(1))
            # else:
            #     membank.to(fine_tune.util.genDevice(0))

            # Move memory bank to `cuda:1`
            membank.to(student_config.device)

        logger.info("Finish membory bank loading")
    except FileNotFoundError as membank_not_found:
        raise FileNotFoundError("Can't load memory bank from file!" +
                "Please run `build_membank.py` first") from membank_not_found

    # Perform layer wise contrastive distillation.
    logger.info("Perform layer wise contrastive distillation")
    fine_tune.util.contrast_distill_layerwise(
        teacher_logitsbank=teacher_logits,
        teacher_device=teacher_config.device,
        student_config=student_config,
        dataset=dataset,
        student_model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        student_tokenizer=student_tokenizer,
        membanks=membanks,
        softmax_temp=args.softmax_temp,
        contrast_temp=args.contrast_temp,
        contrast_steps=args.contrast_steps,
        soft_label_weight=args.soft_label_weight,
        contrast_loss_weight=args.contrast_weight
    )
