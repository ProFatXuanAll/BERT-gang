r"""Build logits bank with given teacher model and dataset.

Usage:
    python build_logitsbank.py ...

Run `python build_logitsbank.py -h` for help, or see documentation for more information.
"""



# built-in modules

import os
import argparse
import logging

# 3rd party modules

import torch

from tqdm import tqdm
# my own modules

import fine_tune

# Get main logger.
logger = logging.getLogger('fine_tune.build_logitsbank')
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

    # Required arguments.
    parser.add_argument(
        '--task',
        help='Name of the distillation task.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--experiment',
        help='Experiment name of the fine-tuned model',
        required=True,
        type=str
    )
    parser.add_argument(
        '--ckpt',
        help='Checkpoint of the model',
        required=True,
        type=int
    )
    parser.add_argument(
        '--model',
        help='Name of the model',
        required=True,
        type=str
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name of the fine-tune task',
        required=True,
        type=str
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

    # Parse arguments.
    args = parser.parse_args()

    # Load fine-tune teacher model configuration.
    config = fine_tune.config.TeacherConfig.load(
        experiment=args.experiment,
        model=args.model,
        task=args.task
    )

    # Change batch size for faster evaluation.
    if args.batch_size:
        config.batch_size = args.batch_size

    # Check user specify device or not.
    if args.device_id > -1:
        config.device_id = args.device_id
    logger.info("Use device: %s to inference", config.device_id)

    # Set dataset.
    config.dataset = args.dataset

    # Log configuration.
    logger.info(config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=config
    )

    dataset = fine_tune.util.load_contrast_dataset_by_config(
        config=config,
        neg_num=1,
        defined_by_label=False
    )

    # Load teacher tokenizer.
    teacher_tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
        config=config
    )

    # Load teacher model from given checkpoint.
    teacher_model = fine_tune.util.load_teacher_model_by_config(
        config=config
    )
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=config.experiment,
        model=config.model,
        task=config.task
    )
    model_name = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name,
        f'model-{args.ckpt}.pt'
    )

    # Load model from checkpoint.
    teacher_model.load_state_dict(torch.load(model_name, map_location=config.device))
    # Init data loader.
    temp_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_collate_fn(),
        num_workers=os.cpu_count(),
        shuffle=False
    )

    # Set teacher model to evaluation mode.
    teacher_model.eval()

    # Init memory bank with teacher representations.
    logger.info("Build logits bank with teacher output logits")

    # Build logits bank.
    logitsbank = fine_tune.model.Logitsbank(
        N = len(dataset),
        C = config.num_class
    )

    for text, text_pair, label, p_indices, _ in tqdm(temp_loader):
        t_batch_encode = teacher_tokenizer(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )

        t_input_ids = t_batch_encode['input_ids']
        t_token_type_ids = t_batch_encode['token_type_ids']
        t_attention_mask = t_batch_encode['attention_mask']

        with torch.no_grad():
            teacher_logits, _ = teacher_model(
                input_ids=t_input_ids.to(config.device),
                token_type_ids=t_token_type_ids.to(config.device),
                attention_mask=t_attention_mask.to(config.device),
                return_hidden_and_attn=False
            )

        logitsbank.update_logits(
            new=teacher_logits.to('cpu'),
            index=torch.LongTensor(p_indices)
        )

    logits_bank_path = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

    fname = os.path.join(logits_bank_path, f'logitsbank-{args.ckpt}.pt')
    logger.info("Save logits bank to: %s", fname)
    torch.save(logitsbank.state_dict(), fname)
