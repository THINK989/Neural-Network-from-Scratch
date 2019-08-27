import numpy as np
"""
    Implementation of the loss function 

    Arguments:
    y_pred -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    loss
    
"""
# Loss function is used to monitor, how the model is performing. 
# It is supposed to approach global optimum i.e. smallest value possible.

def logistic_loss(a3, Y):
    
    """
    Implement the cost function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    cost - value of the cost function
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    return cost

def MAE(y_pred, y):
    cost = np.sum(np.abs(y_pred - y))/m
    return cost

def MSE(y_pred, y):
    cost = np.sum((y_pred - y)**2)/m
    return cost


