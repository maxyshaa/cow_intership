import numpy as np
from sklearn.datasets import load_boston


def data_loader() -> np.ndarray:
    dataset = load_boston()
    x = dataset.data
    y = dataset.target[:, np.newaxis]
    print("Total samples in our dataset is: {}".format(x.shape[0]))
    return x, y





