import torch
import torch.nn as nn
from uncertainty_toolkit.deep_ensemble import DeepEnsemble
from uncertainty_toolkit.visualization import plot_predictions_with_uncertainty

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

models = [SimpleModel() for _ in range(5)]

wrapper = DeepEnsemble(models)

x = torch.linspace(-5, 5, steps=100).unsqueeze(1)
mean, std = wrapper.predict(x)

plot_predictions_with_uncertainty(x.squeeze().numpy(), mean.numpy(), std.numpy())
