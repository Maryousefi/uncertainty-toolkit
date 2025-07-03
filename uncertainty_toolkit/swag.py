import torch
from .base import UncertaintyWrapper

class SWAGWrapper(UncertaintyWrapper):
    def __init__(self, model, swag_samples):
        """
        model: base model architecture
        swag_samples: list of model parameter state_dicts collected during training
        """
        super().__init__(model)
        if not swag_samples:
            raise ValueError("swag_samples list cannot be empty.")
        self.swag_samples = swag_samples

    def predict(self, x, n_samples=30):
        preds = []
        self.model.eval()
        for _ in range(n_samples):
            idx = torch.randint(len(self.swag_samples), (1,)).item()
            self.model.load_state_dict(self.swag_samples[idx])
            with torch.no_grad():
                preds.append(self.model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
