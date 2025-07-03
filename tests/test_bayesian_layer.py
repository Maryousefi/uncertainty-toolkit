import torch
import torch.nn as nn
import pytest
from uncertainty_toolkit.bayesian_layer import BayesianLinear, BayesianWrapper

def test_bayesian_linear_forward_shape():
    layer = BayesianLinear(3, 2)
    x = torch.randn(4, 3)
    out = layer(x)
    assert out.shape == (4, 2), "Output shape should be (batch_size, out_features)"

def test_bayesian_linear_variation():
    layer = BayesianLinear(3, 2)
    x = torch.randn(4, 3)
    out1 = layer(x)
    out2 = layer(x)
    # Outputs should differ due to sampling noise
    assert not torch.allclose(out1, out2), "Outputs should differ because of Bayesian sampling"

def test_bayesian_wrapper_predict_shapes():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bayesian = BayesianLinear(1, 1)

        def forward(self, x):
            return self.bayesian(x)

    model = DummyModel()
    wrapper = BayesianWrapper(model, n_samples=10)
    x = torch.randn(5, 1)
    mean, std = wrapper.predict(x)
    assert mean.shape == (5, 1), "Mean shape mismatch"
    assert std.shape == (5, 1), "Std shape mismatch"
    assert torch.all(std >= 0), "Std must be non-negative"

def test_bayesian_wrapper_predict_consistency():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bayesian = BayesianLinear(1, 1)

        def forward(self, x):
            return self.bayesian(x)

    model = DummyModel()
    wrapper = BayesianWrapper(model, n_samples=5)
    x = torch.randn(3, 1)

    mean1, std1 = wrapper.predict(x)
    mean2, std2 = wrapper.predict(x)

    # Means should differ across predictions due to sampling noise
    assert not torch.allclose(mean1, mean2), "Means should differ between runs"
    # Standard deviations must be positive
    assert torch.all(std1 >= 0)
    assert torch.all(std2 >= 0)
