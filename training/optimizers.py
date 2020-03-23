import numpy as np
from training import compute_cost
from training import data_loader


def gradient_descent(x, y, params: np.ndarray, learning_rate: float, n_iter: Int) -> np.ndarray:
    """Apply a gradient descent as a optimization part.

    Args:
        x: Represents the feature samples.
        y: Represents the target values, labels.
        params: Matrix weight.
        learning_rate: Learning rate.
        n_iter: Number of calculate iterations for the gradient descent.

    Returns:
        Function returns the updated parameter values according the update rule.

        history: History of our costs returned by the cost function in each iteration.
        params: Updated parameter values.

    """
    n_samples = len(y)
    history = np.zeros((n_iter, 1))

    for i in range(n_iter):
        params = params - (learning_rate / n_samples) * x.T @ (x @ params - y)
        history[i] = compute_cost(x, y, params)

    return history, params
