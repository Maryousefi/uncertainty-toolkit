import torch
import torch.nn as nn
from .base import UncertaintyWrapper
from typing import List, Tuple  # ✅ Add Tuple!

class DeepEnsemble(UncertaintyWrapper):
    """
    Deep Ensemble wrapper around multiple trained PyTorch models.

    Args:
        models (List[nn.Module]): List of trained PyTorch models.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__(models[0]) 
        self.models = models

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Predict mean and uncertainty by aggregating predictions from ensemble models.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mean (torch.Tensor): Mean prediction.
            std (torch.Tensor): Standard deviation (uncertainty).
        """
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
