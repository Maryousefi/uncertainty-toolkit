import torch
import numpy as np

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()

def sample_with_replacement(dataset, n_samples):
    """
    Generate a bootstrap sample (with replacement) from a dataset.
    
    Args:
        dataset: a dataset object with __len__ and __getitem__
        n_samples: number of samples in the bootstrap sample
        
    Returns:
        List of sampled data indices
    """
    indices = torch.randint(low=0, high=len(dataset), size=(n_samples,))
    return indices.tolist()
