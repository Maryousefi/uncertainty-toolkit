import torch
import torch.nn as nn
import torch.optim as optim
from uncertainty_toolkit.mc_dropout import MCDropout
from uncertainty_toolkit.visualization import plot_predictions_with_uncertainty

# Example model with Dropout
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Dummy training
model = SimpleModel()


wrapper = MCDropout(model, n_samples=50)

# Example input
x = torch.linspace(-5, 5, steps=100).unsqueeze(1)
mean, std = wrapper.predict(x)

plot_predictions_with_uncertainty(x.squeeze().numpy(), mean.numpy(), std.numpy())
