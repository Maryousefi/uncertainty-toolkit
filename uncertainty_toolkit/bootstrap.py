import torch
from .base import UncertaintyWrapper

class BootstrapEnsemble(UncertaintyWrapper):
    def __init__(self, models):
        """Each model is trained on a bootstrapped dataset."""
        self.models = models

    def predict(self, x):
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
