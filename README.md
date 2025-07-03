# uncertainty_toolkit

**Lightweight PyTorch Toolkit for Uncertainty Quantification in Deep Learning**

---

## Overview

`uncertainty_toolkit` provides a modular and extensible framework to estimate predictive uncertainty in PyTorch models. It implements state-of-the-art methods such as Monte Carlo Dropout, Deep Ensembles, Bayesian Linear Layers, Bootstrap Ensembles, and Stochastic Weight Averaging-Gaussian (SWAG). The toolkit is designed to be lightweight and easy to integrate with existing models, offering uncertainty-aware predictions with minimal code changes.

---

## Features

- **Monte Carlo Dropout**: Approximate Bayesian inference via stochastic forward passes during evaluation.  
- **Deep Ensembles**: Aggregate predictions from multiple independently trained models to estimate epistemic uncertainty.  
- **Bayesian Layers**: Bayesian Linear layers with variational inference for weight uncertainty.  
- **Bootstrap Ensembles**: Ensembles trained on bootstrapped subsets of data to quantify model uncertainty.  
- **SWAG (Stochastic Weight Averaging-Gaussian)**: Efficient posterior approximation by sampling from parameter distributions.  
- **Visualization**: Built-in utilities to visualize predictions with confidence intervals.

---

## Installation

```bash
pip install uncertainty_toolkit
# Or for development (editable install):
git clone https://github.com/Maryousefi/uncertainty-toolkit.git
cd uncertainty-toolkit
pip install -e .
```

---

## Quickstart

```python
import torch
from uncertainty_toolkit import MCDropout, plot_predictions_with_uncertainty

# Wrap your existing PyTorch model with MC Dropout wrapper
mc = MCDropout(model, n_samples=50)

# Perform prediction with uncertainty estimates
mean, std = mc.predict(x)

# Visualize predictions and uncertainty bounds
plot_predictions_with_uncertainty(x.cpu().numpy(), mean.numpy(), std.numpy())
```

---

## Development

To set up the development environment and run tests:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

---

## License

This project is licensed under the MIT License © 2025 Maryam Yousefi
