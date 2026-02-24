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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=1e-3, checkpoint_path=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, f1_macro_avg, model):
        """Monitor F1 macro average and trigger early stopping if needed."""
        score = f1_macro_avg  # Higher F1 is better
        if self.best_score is None:
            self.best_score = score
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
                if self.verbose:
                    print(f'✓ Checkpoint saved (F1: {score:.4f})')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
                if self.verbose:
                    print(f'✓ Checkpoint saved (F1: {score:.4f})')
            self.counter = 0
