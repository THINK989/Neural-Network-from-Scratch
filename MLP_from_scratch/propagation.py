import numpy as np
from activation import *

def forward_propagation(X, parameters):
    """    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    """
    Above code will only work for layers = 4
    Repeat 
    zL = np.dot(WL, aL-1) + bL
    aL = sigmoid(zL)
    
    so that More layers can be propagated through
    """
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) # also add WL, bL, zL, aL if you are adding more layers
    
    return a3, cache

def backward_propagation(X, Y, cache):
    """
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    #Step 1
    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    #Step 2
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    #Step 3
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    """
    Above code will only work for layers = 4
    ADD
    dzL = 1./m * (aL - Y)
    dWL = np.dot(dzL, aL-1.T)
    dbL = np.sum(dzL, axis=1, keepdims = True)
    on top
    and move the rest of the code 1 step down
    
    so that More layers can be propagated through
    """
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1} #Also add dzL, dWL, dbL, daL with their keys look at the dictionary for reference
    
    return gradients











