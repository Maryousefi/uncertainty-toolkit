import torch
from .base import UncertaintyWrapper

class SWAGWrapper(UncertaintyWrapper):
    def __init__(self, model, swag_samples):
        """
        model: base model architecture
        swag_samples: list of model parameter states collected during training
        """
        self.model = model
        self.swag_samples = swag_samples

    def predict(self, x, n_samples=30):
        preds = []
        for i in range(n_samples):
            idx = torch.randint(len(self.swag_samples), (1,)).item()
            self.model.load_state_dict(self.swag_samples[idx])
            self.model.eval()
            with torch.no_grad():
                preds.append(self.model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
