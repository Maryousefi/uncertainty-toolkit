import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_with_uncertainty(x, mean, std, title="Prediction with Uncertainty", save_path=None):
    """
    Plot mean predictions with uncertainty intervals.

    Args:
        x (array-like): 1D array of input points.
        mean (array-like): 1D array of mean predictions.
        std (array-like): 1D array of standard deviations (uncertainties).
        title (str, optional): Plot title. Defaults to "Prediction with Uncertainty".
        save_path (str, optional): If provided, saves the plot to this path instead of showing it.

    Returns:
        matplotlib.figure.Figure: The created matplotlib Figure.
    """
    x = np.array(x).squeeze()
    mean = np.array(mean).squeeze()
    std = np.array(std).squeeze()

    assert x.ndim == 1 and mean.ndim == 1 and std.ndim == 1, "Inputs must be 1D arrays."
    assert len(x) == len(mean) == len(std), "x, mean, and std must have the same length."

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean, label="Mean Prediction")
    ax.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.3, label="95% Confidence Interval")
    ax.set_title(title)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

    return fig
