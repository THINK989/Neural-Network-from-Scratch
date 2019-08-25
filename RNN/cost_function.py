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

def Logistic_loss(y_pred, y):
    cost = -np.sum(np.multiply(y,np.log(y_pred)) + np.multiply((1-y),np.log(1-y_pred))) / m
    return np.squeeze(cost)

def MAE(y_pred, y):
    cost = np.sum(np.abs(y_pred - y))/m
    return cost

def MSE(y_pred, y):
    cost = np.sum((y_pred - y)**2)/m
    return cost


