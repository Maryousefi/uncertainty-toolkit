import torch
import torch.nn as nn
from uncertainty_toolkit.bayesian_layer import BayesianLinear

class BayesianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = BayesianLinear(1, 32)
        self.out = BayesianLinear(32, 1)

    def forward(self, x):
        x = torch.relu(self.b1(x))
        return self.out(x)

# Example usage:
model = BayesianNet()
x = torch.randn(10, 1)
y = model(x)
print(y)
