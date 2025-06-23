import torch
import numpy as np
import random

def set_seed(seed):
    """Sets the random seed for reproducibility across different libraries."""
    # 1. Python's built-in random module
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch on GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    
    # 5. cuDNN (for deterministic convolution algorithms)
    # This might make some operations slower, but ensures determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Disable benchmarking to avoid non-determinism

    print(f"Random seed set to {seed}")