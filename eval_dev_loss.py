r"""This script is used to log the validation loss.
Validation loss = soft target + hard target loss on validation set.
We need to load a teacher model with given checkpoint to generate soft target,
and load a student model with several checkpoint to calculate the loss.
Note:
    It is a temporary script!
"""

# built-in modules

import argparse
import logging
import os
import re

# typing
from typing import List

# 3rd-party modules

import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import transformers

from tqdm import tqdm

# my own modules

import fine_tune

@torch.no_grad()
def get_loss(
    tconfig: fine_tune.config.TeacherConfig,
    sconfig: fine_tune.config.StudentConfig,
    data: fine_tune.task.Dataset,
    tmodel: fine_tune.model.TeacherModel,
    smodel: fine_tune.model.StudentModel,
    teacher_tknr: transformers.PreTrainedTokenizer,
    student_tknr: transformers.PreTrainedTokenizer,
    log_step: int,
    alpha: float = 0.2,
    gamma: float = 0.8,
    softmax_temp: float = 1.0
) -> List[int]:
    """Get distill loss with evaluation dataset.

    Parameters
    ----------
    tconfig : fine_tune.config.TeacherConfig
        `fine_tune.config.TeacherConfig` class which attributes are used
        for experiment setup.
    sconfig : fine_tune.config.StudentConfig
        `fine_tune.config.StudentConfig` class which attributes are used
        for experiment setup.
    data : fine_tune.task.Dataset
        Task specific dataset.
    tmodel : fine_tune.model.TeacherModel
        A fine-tuned teacher model which is used to
        generate soft targets, hidden states and attentions.
    smodel : fine_tune.model.StudentModel
        A distilled student model to generate task-relative prediction.
    teacher_tknr : transformers.PreTrainedTokenizer
        Tokenizer paired with `teacher_model`.
    student_tknr : transformers.PreTrainedTokenizer
        Tokenizer paired with `student_model`.
    log_step : int
        Logging interval.
    alpha : float, optional
        Weight of soft target loss, by default 0.2
    gamma: float, optional
        Weight of hard target loss, by default 0.8
    softmax_temp : float, optional
        Softmax temperature, by default 1.0

    Returns
    -------
    List[int]
        List of batch loss value.
    """
    # Set to evaluation mode.
    tmodel.eval()
    smodel.eval()

    # Model running device of teacher and student model.
    tdevice = tconfig.device
    sdevice = sconfig.device

    # Construct data loader.
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=tconfig.batch_size,
        collate_fn=data.create_collate_fn(),
        num_workers=os.cpu_count(),
        shuffle=False
    )

    # Get loss function.
    objective = fine_tune.objective.distill_loss

    # Batch loss placeholder.
    loss = 0
    losses = []

    # Mini-batch loop.
    for step, (text, text_pair, label) in enumerate(tqdm(dataloader)):
        # Transform `label` to a Tensor.
        label = torch.LongTensor(label)

        # Get `input_ids`, `token_type_ids` and `attention_mask` via tokenizer.
        teacher_batch_encode = teacher_tknr(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=teacher_config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        teacher_input_ids = teacher_batch_encode['input_ids']
        teacher_token_type_ids = teacher_batch_encode['token_type_ids']
        teacher_attention_mask = teacher_batch_encode['attention_mask']

        student_batch_encode = student_tknr(
            text=text,
            text_pair=text_pair,
            padding='max_length',
            max_length=student_config.max_seq_len,
            return_tensors='pt',
            truncation=True
        )
        student_input_ids = student_batch_encode['input_ids']
        student_token_type_ids = student_batch_encode['token_type_ids']
        student_attention_mask = student_batch_encode['attention_mask']

        teacher_logits, _, _ = tmodel(
            input_ids = teacher_input_ids.to(tdevice),
            token_type_ids=teacher_token_type_ids.to(tdevice),
            attention_mask=teacher_attention_mask.to(tdevice),
            return_hidden_and_attn=True
        )

        # Get output logits, hidden states and attentions from student.
        student_logits, _, _ = smodel(
            input_ids = student_input_ids.to(sdevice),
            token_type_ids=student_token_type_ids.to(sdevice),
            attention_mask=student_attention_mask.to(sdevice),
            return_hidden_and_attn=True
        )

        loss = objective(
            hard_target=label.to(sdevice),
            teacher_logits=teacher_logits.to(sdevice),
            student_logits=student_logits,
            gamma=gamma,
            alpha=alpha,
            softmax_temp=softmax_temp
        ).item()

        if step % log_step == 0:
            losses.append(loss)

    return losses

# Get main logger.
logger = logging.getLogger('fine_tune.eval_dev_loss')
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
        help='Name of the student experiment to evalutate loss.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--texperiment',
        help='Name of the teacher experiment to generate soft target.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--model',
        help='Name of the model to evaluate loss.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tmodel',
        help='Name of the teacher model to generate soft target.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tckpt',
        help='checkpoint of teacher model.',
        required=True,
        type=int
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
        '--log_step',
        help='Logging interval.',
        required=True,
        type=int,
    )

    # Optional parameters.
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
        '--batch_size',
        default=32,
        help='Evaluation batch size.',
        type=int,
    )
    parser.add_argument(
        '--tdevice_id',
        default=-1,
        help='Load teacher model on dedicated device.',
        type=int,
    )
    parser.add_argument(
        '--device_id',
        default=-1,
        help='Load student model on dedicated device.',
        type=int,
    )
    parser.add_argument(
        '--start_ckpt',
        default=0,
        help='Start evaluation from specified checkpoint',
        type=int
    )

    # Parse arguments.
    args = parser.parse_args()

    # Load teacher and student model config.
    teacher_config = fine_tune.config.TeacherConfig.load(
        experiment=args.texperiment,
        model=args.tmodel,
        task=args.task
    )

    student_config = fine_tune.config.StudentConfig.load(
        experiment=args.experiment,
        model=args.model,
        task=args.task
    )

    # Change batch size for faster evaluation.
    if args.batch_size:
        teacher_config.batch_size = args.batch_size
        student_config.batch_size = args.batch_size

    # Check user specify device or not.
    if args.device_id > -1:
        student_config.device_id = args.device_id
    if args.tdevice_id > -1:
        teacher_config.device_id = args.tdevice_id
    logger.info("Load teacher model to device: %s", teacher_config.device_id)
    logger.info("Load student model to device: %s", student_config.device_id)

    # Set evaluation dataset.
    teacher_config.dataset = args.dataset
    student_config.dataset = args.dataset

    # Log configuration
    logger.info(teacher_config)
    logger.info(student_config)

    # Control random seed for reproducibility.
    fine_tune.util.set_seed_by_config(
        config=teacher_config
    )

    # Load validation/development dataset.
    dataset = fine_tune.util.load_dataset_by_config(
        config=teacher_config
    )

    # Load teacher tokenizer and model.
    teacher_tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(
        config=teacher_config
    )
    teacher_model = fine_tune.util.load_teacher_model_by_config(
        config=teacher_config
    )

    # Load student tokenizer and model.
    student_tokenizer = fine_tune.util.load_student_tokenizer_by_config(
        config=student_config
    )
    student_model = fine_tune.util.load_student_model_by_config(
        config=student_config,
        tokenizer=student_tokenizer
    )

    # Load teacher model from checkpoint.
    texp_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=teacher_config.experiment,
        model=teacher_config.model,
        task=teacher_config.task
    )
    tmodel_ckpt_path = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        texp_name,
        f'model-{args.tckpt}.pt'
    )
    logger.info("Load teacher model from %s", tmodel_ckpt_path)
    teacher_model.load_state_dict(
        torch.load(
            tmodel_ckpt_path,
            map_location=teacher_config.device
        )
    )

    # Get all student model checkpoint file names.
    experiment_name = fine_tune.config.BaseConfig.experiment_name(
        experiment=student_config.experiment,
        model=student_config.model,
        task=student_config.task
    )
    experiment_dir = os.path.join(
        fine_tune.path.FINE_TUNE_EXPERIMENT,
        experiment_name
    )

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
    all_ckpts = list(filter(lambda ckpt: ckpt >= args.start_ckpt, all_ckpts))

    # Create tensorboard's `SummaryWriter`.
    writer = torch.utils.tensorboard.SummaryWriter(
        os.path.join(
            fine_tune.path.LOG,
            experiment_name
        )
    )

    # Use a list to store batch loss.
    losses = []

    for ckpt in all_ckpts:
        logger.info("Load from ckpt: %s", ckpt)

        # Clean all gradient.
        student_model.zero_grad()

        # Load model from checkpoint.
        student_model.load_state_dict(
            torch.load(
                os.path.join(experiment_dir, f'model-{ckpt}.pt'),
                map_location=student_config.device
            )
        )

        losses.extend(
            get_loss(
                tconfig=teacher_config,
                sconfig=student_config,
                data=dataset,
                tmodel=teacher_model,
                smodel=student_model,
                teacher_tknr=teacher_tokenizer,
                student_tknr=student_tokenizer,
                log_step=args.log_step,
                alpha=args.soft_weight,
                gamma=args.hard_weight,
                softmax_temp=args.softmax_temp
            )
        )

    # Write to tensorboard.
    for step, loss in enumerate(losses):
        writer.add_scalar(
            f'{student_config.task}/{student_config.dataset}/{student_config.model}'+
            '/loss',
            loss,
            step
        )
