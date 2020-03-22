#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston


# define our cost function

def compute_cost(X, y, params):
    n_samples = len(y)
    h = X @ params
    return (1/(2*n_samples))*np.sum((h-y)**2)


def gradient_descent(X, y, params, learning_rate, n_iters):
    n_samples = len(y)
    J_history = np.zeros((n_iters,1))
    print('params shape ', (X.T).shape)

    for i in range(n_iters):
        params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
        J_history[i] = compute_cost(X, y, params)

    return J_history, params


def linear_regression(X, y, n_iter=1500, lr=0.05):
    # lr= learning_rate
    
    n_samples = len(y)
    
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X-mu) / sigma
    
    X = np.hstack((np.ones((n_samples,1)),X))
    n_features = np.size(X,1)
    params = np.zeros((n_features,1))
    
    initial_cost = compute_cost(X, y, params)
    print("Initial cost is: ", initial_cost, "\n")
    (J_history, optimal_params) = gradient_descent(X, y, params, learning_rate, n_iters)
    print("Optimal parameters are: \n", optimal_params, "\n")
    print("Final cost is: ", J_history[-1])
    plt.plot(range(len(J_history)), J_history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    
    
dataset = load_boston()
X = dataset.data
y = dataset.target[:,np.newaxis]

linear_regression(X, y)