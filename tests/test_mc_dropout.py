import torch
import torch.nn as nn
import pytest
from uncertainty_toolkit.mc_dropout import MCDropout

# Simple dummy model for testing :)
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def test_mc_dropout_predict_shapes():
    model = DummyModel()
    mc = MCDropout(model, n_samples=10)
    
    x = torch.randn(5, 1)
    mean, std = mc.predict(x)
    
    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert torch.all(std >= 0), "Standard deviation must be non-negative"

def test_mc_dropout_output_consistency():
    model = DummyModel()
    mc = MCDropout(model, n_samples=5)
    x = torch.randn(3, 1)
    
    # Predict twice, should be close but not necessarily identical
    mean1, std1 = mc.predict(x)
    mean2, std2 = mc.predict(x)
    
    # Means should not be exactly the same due to MC dropout randomness
    assert not torch.allclose(mean1, mean2)
    
    # should be positive
    assert torch.all(std1 >= 0)
    assert torch.all(std2 >= 0)
