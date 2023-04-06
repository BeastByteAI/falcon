try:
    from matplotlib import pyplot as plt
except (ImportError, ModuleNotFoundError):
    plt = None
import numpy as np

def _plot_errors(train_data: np.ndarray, test_data: np.ndarray, pred: np.ndarray) -> None: 
    if plt is None:
        print("matplotlib is not installed")
        return None
    window_size = len(test_data) - len(pred)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_data)), train_data, label="train")
    plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data,  label="test", alpha = 0.75)
    plt.plot(np.arange(len(train_data) + window_size, len(train_data) + len(test_data)), pred,  label="forecast", alpha = 0.75)
    plt.legend()
    plt.show()