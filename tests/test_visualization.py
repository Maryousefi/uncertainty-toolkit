import numpy as np
import matplotlib.pyplot as plt
from uncertainty_toolkit.visualization import plot_predictions_with_uncertainty

def test_plot():
    x = np.linspace(-5, 5, 100)
    mean = np.sin(x)
    std = np.ones_like(x) * 0.1

    fig = plot_predictions_with_uncertainty(x, mean, std)
    
    assert fig is not None
    plt.close(fig)  # Close figure to free memory
