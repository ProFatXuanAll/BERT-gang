r"""Note: It is a temporary file
Use this file to load a trained distil-bert model to evaluate.
"""
#TODO: remove this file.

# built-in modules.

import argparse
import logging
import os
import re

import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.utils.tensorboard
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# 3rd party modules.
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

import fine_tune

# my own modules


# Get main logger.
logger = logging.getLogger('fine_tune.distill_bert_eval')
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

    # Set `torch.device`.
    if args.device_id == -1:
        device = torch.device('cpu')
    elif args.device_id >= 0:
        device = torch.device(f'cuda:{args.device_id}')
    else:
        raise ValueError(f"Invalid device id : {args.device_id}")

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

    # Create tensorboard's `SummaryWriter`
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
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
                os.path.join(experiment_dir, f'model-{ckpt}.pt'),
                map_location=device
            )
        )
        logger.info("Load model from ckpt %s", ckpt)
        model.eval()

        # Calculate accuracy
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.create_collate_fn(),
            shuffle=False
        )

        # Record label and prediction for calculating accuracy.
        all_label = []
        all_pred_label = []

        # Evaluate through mini-batch loop.
        mini_batch_iterator = tqdm(dataloader)

        for text, text_pair, label in mini_batch_iterator:
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

            # Mini-batch prediction.
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
            pred_label = F.softmax(
                outputs.logits,
                dim=-1
            ).argmax(dim=-1).to('cpu')

            all_label.extend(label)
            all_pred_label.extend(pred_label.tolist())

        # Calculate accuracy.
        acc = accuracy_score(all_label, all_pred_label)

        # Show accuracy.
        mini_batch_iterator.set_description(f'accuracy: {acc:.6f}')

        # Realease IO resources.
        mini_batch_iterator.close()

        # Update max accuracy.
        if max_acc <= acc:
            max_acc = acc
            max_acc_ckpt = ckpt

        # Log accuracy.
        writer.add_scalar(
            f'{args.task}/{args.dataset}/accuracy',
            acc,
            ckpt
        )

    # Release IO resources.
    writer.flush()
    writer.close()

    # Log maximum accuracy.
    logger.info('max accuracy:            %f', max_acc)
    logger.info('max accuracy checkpoint: %d', max_acc_ckpt)


