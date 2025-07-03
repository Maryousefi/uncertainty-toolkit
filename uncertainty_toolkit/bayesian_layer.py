import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import UncertaintyWrapper

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

    def forward(self, input):
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_eps * torch.exp(0.5 * self.weight_logvar)
        bias = self.bias_mu + bias_eps * torch.exp(0.5 * self.bias_logvar)

        return F.linear(input, weight, bias)

class BayesianWrapper(UncertaintyWrapper):
    def __init__(self, model, n_samples=20):
        super().__init__(model)
        self.n_samples = n_samples

    def predict(self, x):
        preds = []
        for _ in range(self.n_samples):
            preds.append(self.model(x).detach().cpu())
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
