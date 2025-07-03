import torch
import torch.nn as nn
from uncertainty_toolkit.deep_ensemble import DeepEnsemble

def test_deep_ensemble_shape():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    models = [DummyModel(), DummyModel()]
    wrapper = DeepEnsemble(models)
    x = torch.randn(5, 1)
    mean, std = wrapper.predict(x)

    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
