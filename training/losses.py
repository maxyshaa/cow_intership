import numpy as np
from training import data_loader


def compute_cost(x, y, params: np.ndarray) -> np.ndarray:
    """ Define a mean squared loss function.

    Args:
        x: Represents the feature samples.
        y: Represents the target values, labels.
        params: Matrix parameters (weight).

    Returns:


    """
    n_samples = len(y)
    h = x @ params

    return (1 / (2 * n_samples)) * np.sum((h - y) ** 2)
