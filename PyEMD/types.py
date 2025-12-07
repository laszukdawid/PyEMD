from typing import Union

import numpy as np

EmdArray = np.ndarray
EmdDType = np.dtype

try:
    import torch

    EmdArray = Union[np.ndarray, torch.Tensor]
    EmdDType = Union[np.dtype, torch.dtype]
except ImportError:
    pass
