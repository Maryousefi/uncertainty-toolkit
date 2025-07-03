import numpy as np
from uncertainty_toolkit.visualization import plot_predictions_with_uncertainty

def test_plot():
    x = np.linspace(-5, 5, 100)
    mean = np.sin(x)
    std = np.ones_like(x) * 0.1

    # Call the function and store the returned Figure object
    fig = plot_predictions_with_uncertainty(x, mean, std)

    # Check that a valid Figure was returned
    assert fig is not None
