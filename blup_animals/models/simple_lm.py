def linear_regression(x, y, n_iter=1500, lr=0.05):
    
    n_samples = len(y)
    
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x = (x-mu) / sigma
    
    x = np.hstack((np.ones((n_samples,1)),x))
    n_features = np.size(x,1)
    params = np.zeros((n_features,1))
    
    initial_cost = compute_cost(x, y, params)
    print("Initial cost is: ", initial_cost, "\n")
    (history, optimal_params) = gradient_descent(x, y, params, learning_rate, n_iters)
    print("Optimal parameters are: \n", optimal_params, "\n")
    print("Final cost is: ", history[-1])
    plt.plot(range(len(history)), history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    
  
