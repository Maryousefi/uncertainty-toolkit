import torch
from .base import UncertaintyWrapper
from typing import List
import torch.nn as nn

class BootstrapEnsemble(UncertaintyWrapper):
    """
    Bootstrap ensemble wrapper around multiple models trained on bootstrapped datasets.

    Args:
        models (List[nn.Module]): List of trained PyTorch models.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__(models[0])  
        self.models = models

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and uncertainty by aggregating predictions from all models.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mean (torch.Tensor): Mean prediction across models.
            std (torch.Tensor): Standard deviation (uncertainty) across models.
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
