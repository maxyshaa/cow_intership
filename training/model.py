import numpy as np
from training import compute_cost
from training import fradient_descent


def linear_regression(x, y: np.ndarray, n_iter=1500, learning_rate=0.05) -> None:
    """Compute the linear regression model.

    Args:
        x: Represents the feature samples.
        y: Represents the target values, labels.
        n_iter: Number of calculate iterations for the gradient descent. Defaults to 1500.
        learning_rate: Learning rate default to 0.05.
    """
    n_samples = len(y)
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x = (x - mu) / sigma

    x = np.hstack((np.ones((n_samples, 1)), x))
    n_features = np.size(x, 1)
    params = np.zeros((n_features, 1))

    initial_cost = compute_cost(x, y, params)
    print("Initial cost is: ", initial_cost, "\n")
    (history, optimal_params) = gradient_descent(x, y, params, learning_rate, n_iters)
    print("Optimal parameters are: \n", optimal_params, "\n")
    print("Final cost is: ", history[-1])
    

