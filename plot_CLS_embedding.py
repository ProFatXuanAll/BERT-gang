r"""Plot [CLS] embeddings to tensorboard projector

Usage:
    python plot_embedding.py ...

Run `python plot_embedding.py -h` for help, or see 'doc/fine_tune_*.md' for
more information.
"""

# built-in modules

import argparse
import logging
import os

# 3rd-party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard

from tqdm import tqdm

# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.eval')
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
    parser.add_argument(
        '--ckpt',
        help='Checkpoint number.',
        required=True,
        type=str
    )
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
        help='Evaluation batch size.',
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
        logger.info("Use teacher model")
    # Load fine-tune distillation student model configuration.
    except TypeError:
        config = fine_tune.config.StudentConfig.load(
            experiment=args.experiment,
            model=args.model,
            task=args.task
        )
        logger.info("Use student model")

    # Change batch size for faster evaluation.
    if args.batch_size:
        config.batch_size = args.batch_size

    # Check user specify device or not.
    if args.device_id > -1:
        config.device_id = args.device_id
    logger.info("Use device: %s to plot CLS embedding", config.device_id)

    # Set evaluation dataset.
    config.dataset = args.dataset

    # Log configuration.
    logger.info(config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    # Load fine-tune / distillation dataset.
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

    # Get experiment name and path.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Load model from checkpoint.
    logger.info("Load model from checkpoint: %s", args.ckpt)
    model.load_state_dict(torch.load(os.path.join(
        experiment_dir,
        f'model-{args.ckpt}.pt'
    )))

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        shuffle=False
    )

    # Inference through mini-batch loop.
    mini_batch_iterator = tqdm(dataloader)

    # Record all labels and CLS embeddings
    all_label = []
    all_CLS = None

    with torch.no_grad():
        for steps, (text, text_pair, label) in enumerate(mini_batch_iterator):
            # Get `input_ids`, `token_type_ids` and `attention_mask` via tokenizer.
            batch_encode = tokenizer(
                text=text,
                text_pair=text_pair,
                padding='max_length',
                max_length=config.max_seq_len,
                return_tensors='pt',
                truncation=True
            )
            input_ids = batch_encode['input_ids']
            token_type_ids = batch_encode['token_type_ids']
            attention_mask = batch_encode['attention_mask']

            # Mini-batch inference
            _, CLS = model(
                input_ids = input_ids.to(config.device),
                token_type_ids = token_type_ids.to(config.device),
                attention_mask = attention_mask.to(config.device)
            )

            # extract CLS of last layer
            CLS = CLS.to('cpu')
            if steps == 0:
                all_CLS = CLS
            else:
                all_CLS = torch.cat((all_CLS, CLS), dim=0)

            # record labels
            all_label.extend(label)

            # Update CLI logger.
            mini_batch_iterator.set_description(f"Iteration: {steps+1}")

    # Update CLI logger.
    mini_batch_iterator.set_description("CLS embeddings generation...done!")

    # Release IO resources.
    mini_batch_iterator.close()

    # Add embeddings to tensorboard projector.
    logger.info("Plot [CLS] embeddings to tensorboard projector")
    writer.add_embedding(mat=all_CLS, metadata=all_label)
    logger.info("Done!")
