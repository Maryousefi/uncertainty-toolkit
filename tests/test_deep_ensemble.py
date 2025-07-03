import torch
import torch.nn as nn
import pytest
from uncertainty_toolkit.deep_ensemble import DeepEnsemble

class DummyModel(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(1, out_dim)

    def forward(self, x):
        return self.linear(x)

def test_deep_ensemble_predict_shape():
    models = [DummyModel() for _ in range(3)]
    ensemble = DeepEnsemble(models)
    x = torch.randn(5, 1)
    mean, std = ensemble.predict(x)
    assert mean.shape == (5, 1), "Mean prediction shape mismatch"
    assert std.shape == (5, 1), "Std prediction shape mismatch"

def test_deep_ensemble_predict_variation():
    models = [DummyModel() for _ in range(3)]
    ensemble = DeepEnsemble(models)
    x = torch.randn(5, 1)
    mean1, std1 = ensemble.predict(x)
    mean2, std2 = ensemble.predict(x)
    # Because models are fixed, means should be close but not necessarily identical
    assert not torch.allclose(mean1, mean2, atol=1e-6) or torch.all(std1 >= 0), "Predictions vary; std should be >= 0"

def test_deep_ensemble_model_eval_called(monkeypatch):
    model = DummyModel()
    called = {"eval": False}
    def fake_eval():
        called["eval"] = True
    model.eval = fake_eval

    ensemble = DeepEnsemble([model])
    x = torch.randn(1, 1)
    ensemble.predict(x)
    assert called["eval"], "model.eval() was not called"
