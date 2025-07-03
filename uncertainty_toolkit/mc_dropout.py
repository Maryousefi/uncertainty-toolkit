import torch
import torch.nn as nn
from .base import UncertaintyWrapper

class MCDropout(UncertaintyWrapper):
    def __init__(self, model, n_samples=20):
        super().__init__(model)
        self.n_samples = n_samples
        self._enable_dropout()

    def _enable_dropout(self):
        """Ensure all dropout layers are active during inference."""
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        self.model.apply(apply_dropout)

    def predict(self, x):
        """Run multiple stochastic forward passes."""
        preds = []
        for _ in range(self.n_samples):
            preds.append(self.model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
