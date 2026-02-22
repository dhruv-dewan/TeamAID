import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    # 1. Basic Python and NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # 3. HPC/CUDA Specifics (The "Deterministic" flags)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. Force CPU/GPU algorithm determinism (latest PyTorch versions)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Seed set to {seed} (Deterministic mode enabled)")