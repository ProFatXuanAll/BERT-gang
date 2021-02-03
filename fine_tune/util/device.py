r"""Utility function to generate torch DEVICE string.
"""

import torch

def genDevice(device_id: int =-1) -> torch.device:
    """Given device id, return relative `torch.device` object

    Parameters
    ----------
    device_id : int, optional
        device id, `-1` means `CPU`, `0` means `cuda:0` and so on., by default -1

    Returns
    -------
    torch.device
        torch device object

    Raises
    ------
    ValueError
        Given unsupported device id
    """
    if device_id == -1:
        return torch.device('cpu')

    if device_id > ( torch.cuda.device_count() - 1 ) :
        raise ValueError("Insufficient cuda device!")
    return torch.device(f'cuda:{device_id}')
