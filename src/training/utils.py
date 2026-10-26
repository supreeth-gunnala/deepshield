"""
DeepShield Training Utilities Module
Provides seed setting for reproducibility and training environment setup.
"""

import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, PyTorch CPU and CUDA to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] Set global random seed = {seed}")
