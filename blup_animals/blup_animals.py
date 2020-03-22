# File containing architecture of animal model

# The model is y_{i,j} = p_{i} + a_{j} + e_{i,j}

# where y_{i,j} is the pre-weaning gain of jth calf
# of the ith sex;
# p_{i} is the fixed effect of ith sex;
# a_{j} is random effect of the ith calf;
# e_{i,j} is a random error effect;

# The goal of model is to estimate the effects of sex
# and predict breeding values for all animals


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constructing the mixed model equations

# import data

def data_load(name):
    # df = pd.read_csv("name")
    # X = df.Sex.map({"Male": 1, "Female": 0})
    # Z = 
    # y = data.loc[: , ["WWG (kg)"]]
    return None


# define our cost function

def compute_cost(X, y, params):
    # h = X @ params
    # return (1/(2*n_samples))*np.sum((h-y)**2)
    return None

def gradient_descent(X, y, params, learning_rate, n_iters):
    n_samples = len(y)
    J_history = np.zeros((n_iters,1))
    print('params shape ', (X.T).shape)

    for i in range(n_iters):
        params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
        J_history[i] = compute_cost(X, y, params)

    return J_history, params


def linear_regression(X, y, n_iter=1500, lr=0.05):
  return None
    
  
