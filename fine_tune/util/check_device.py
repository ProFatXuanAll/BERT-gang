r""" Warning! It is a temporary module!
Use this module during multi-GPU distillation to check GPU numbers.
"""
import torch

def check_device() -> bool:
    r"""
    Check number of GPU during multi-GPU distillation
    """

    gpu_count = torch.cuda.device_count()
    if  not ( torch.cuda.is_available() and gpu_count == 2 ):
        raise ValueError(f'Insufficient GPU device, GPU count: {gpu_count}')
