import matplotlib.pyplot as plt

def plot_predictions_with_uncertainty(x, mean, std, title="Prediction with Uncertainty"):
    """
    x: input points (1D)
    mean: mean predictions (1D)
    std: standard deviation (1D)
    """
    mean = mean.squeeze()
    std = std.squeeze()

    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, label="Mean Prediction")
    plt.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.3, label="95% Confidence Interval")
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()
