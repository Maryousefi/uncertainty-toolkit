import torch
import torch.nn as nn
from uncertainty_toolkit.mc_dropout import MCDropout

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(self.dropout(x))

def test_mc_dropout_output_consistency():
    model = DummyModel()
    mc = MCDropout(model, n_samples=5)
    x = torch.randn(3, 1)

    mean1, std1 = mc.predict(x)
    mean2, std2 = mc.predict(x)

    # Means should not be exactly the same due to MC dropout randomness
    assert not torch.allclose(mean1, mean2)
