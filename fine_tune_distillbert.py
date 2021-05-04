r"""Note: It is a temporary file
Use this file to fine-tune `distil-bert-uncased` on GLUE dataset.
After fine-tuned a distil-bert to downstream task,
we can load it from checkpoint to perform probing task.
"""
#TODO: remove this file.

# built-in modules

import argparse
import logging
import os
import json

# 3rd party modules.
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.distill_bert_train')
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

    # Required parameters
    parser.add_argument(
        '--experiment',
        help='Name of the current fine-tune experiment.',
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
    parser.add_argument(
        '--num_class',
        default=2,
        help='Number of classes to classify.',
        required=True,
        type=int,
    )

    # Optional arguments.
    parser.add_argument(
        '--accum_step',
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
        '--ckpt_step',
        default=1000,
        help='Checkpoint save interval.',
        type=int,
    )
    parser.add_argument(
        '--device_id',
        help='Device ID of model.',
        default=0,
        type=int,
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
        '--seed',
        default=42,
        help='Control random seed.',
        type=int,
    )
    parser.add_argument(
        '--total_step',
        default=50000,
        help='Total number of step to perform training.',
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

    # Parse arguments.
    args = parser.parse_args()

    # Set `torch.device`.
    if args.device_id == -1:
        device = torch.device('cpu')
    elif args.device_id >= 0:
        device = torch.device(f'cuda:{args.device_id}')
    else:
        raise ValueError(f"Invalid device id : {args.device_id}")

    # Control random seed for reproducibility.
    fine_tune.util.set_seed(args.device_id, args.seed)

    # Load fine-tune dataset.
    dataset = fine_tune.util.load_dataset(
        dataset=args.dataset,
        task=args.task
    )

    # Get experiment name and path.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=args.experiment,
        model='bert',
        task=args.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # Store hyper parameters.
    hparams = {
        'experiment': args.experiment,
        'task': args.task,
        'dataset': args.dataset,
        'num_class': args.num_class,
        'accum_step': args.accum_step,
        'batch_size': args.batch_size,
        'ckpt_step': args.ckpt_step,
        'device_id': args.device_id,
        'log_step': args.log_step,
        'lr': args.lr,
        'max_norm': args.max_norm,
        'seed': args.seed,
        'total_step': args.total_step,
        'warmup_step': args.warmup_step,
        'weight_decay': args.weight_decay
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(hparams, f)

    # Create tensorboard's `SummaryWriter`
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Construct dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // args.accum_step,
        collate_fn=dataset.create_collate_fn(),
        num_workers=os.cpu_count(),
        shuffle=True
    )

    # Construct tokenizer.
    logger.info("Load pre-trained distil BERT tokenizer")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Construct model.
    logger.info("Load pre-trained distil BERT model")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        return_dict = True
    ).to(device)

    # Init optimizer.
    logger.info("Load AdamW optimizer")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
    )

    # Init scheduler.
    logger.info("Load linear scheduler")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=args.total_step,
        num_warmup_steps=args.warmup_step
    )

    # Set model to training mode.
    model.train()

    # Clean all gradient.
    optimizer.zero_grad()

    # Step and accumulation step counter.
    step = 0
    accum_step = 0
    total_accum_step = args.total_step * args.accum_step

    # Mini-batch loss and accumulate loss.
    # Update when accumulate to `config.batch_size`.
    loss = 0
    accum_loss = 0

    # `tqdm` CLI Logger. We will manually update progress bar.
    cli_logger = tqdm(
        desc=f'loss: {loss:.6f}',
        total=args.total_step
    )

    # Start training loop.
    logger.info("Start training")

    # Total update times: `args.total_step`.
    while accum_step < total_accum_step:
        for text, text_pair, label in dataloader:
            label = torch.LongTensor(label)

            batch_encode = tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
                truncation=True
            )
            input_ids = batch_encode['input_ids']
            attention_mask = batch_encode['attention_mask']

            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=label.to(device)
            )

            accum_loss = outputs.loss / args.accum_step

            # Mini-batch cross-entropy loss. Only used as log.
            loss += accum_loss.item()

            # Backward pass accumulation loss.
            accum_loss.backward()

            # Increment accumulation step.
            accum_step += 1

            # Perform gradient descend when achieve actual mini-batch.
            if accum_step % args.accum_step == 0:
                # Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_norm
                )

                # Gradient descend.
                optimizer.step()

                # Updaet learning rate.
                scheduler.step()

                # Log on CLI.
                cli_logger.update()
                cli_logger.set_description(
                    f'loss: {loss:.6f}'
                )

                # Increment actual step.
                step += 1

                # Log loss and learning rate for each `args.log_step`.
                if step % args.log_step == 0:
                    writer.add_scalar(
                        f'{args.task}/{args.dataset}/loss',
                        loss,
                        step
                    )
                    writer.add_scalar(
                        f'{args.task}/{args.dataset}/lr',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        step
                    )

                # Clean up mini-batch loss.
                loss = 0

                # Clean up gradient.
                optimizer.zero_grad()

                # Save model for each `args.ckpt_step`.
                if step % args.ckpt_step == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(experiment_dir, f'model-{step}.pt')
                    )

            # Stop training condition.
            if accum_step >= total_accum_step:
                break

    # Release IO resources.
    writer.flush()
    writer.close()
    cli_logger.close()
