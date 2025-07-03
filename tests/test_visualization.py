import numpy as np
from uncertainty_toolkit.visualization import plot_predictions_with_uncertainty

def test_plot():
    x = np.linspace(-5, 5, 100)
    mean = np.sin(x)
    std = np.ones_like(x) * 0.1
    plot_predictions_with_uncertainty(x, mean, std)
