import torch

class UncertaintyWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        raise NotImplementedError("Implement in subclass.")
    
class UncertaintyWrapper:
    """
    Base class for uncertainty estimation wrappers.

    Args:
        model (torch.nn.Module): The PyTorch model to wrap.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        """
        Predict output and uncertainty estimates for input tensor x.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: (mean predictions, uncertainty estimates)
        """
        raise NotImplementedError("Implement in subclass.")
