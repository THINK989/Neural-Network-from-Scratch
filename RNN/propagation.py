import numpy as np
from activation import *

def forward(X, parameters, activation):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)],A_prev) + parameters['b' + str(l)]
        cache_bactivation = (A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        if activation == 'relu':
            A,cache_afteractivation = RELU(Z)  
        cache = (cache_bactivation, cache_afteractivation)
        caches.append(cache)
        
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    Z = np.dot(parameters['W' + str(L)],A) + parameters['b' + str(L)]
    cache_bactivation = (A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL,cache_afteractivation = Sigmoid(Z)  
    cache = (cache_bactivation, cache_afteractivation)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
    


def backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]

    linear_cache, activation_cache = caches[L - 1]
    dZ = sigmoid_backward(dA, activation_cache)
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    grads["dW" + str(L)] = (1/m) * np.dot(dZ, linear_cache[0].T)
    grads["db" + str(L)] = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    grads["dA" + str(L)] = np.dot(linear_cache[1].T, dZ)

    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        linear_cache, activation_cache = caches[l]
        dZ = relu_backward(dA, activation_cache)
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        grads["dW" + str(l+1)] = (1/m) * np.dot(dZ, linear_cache[0].T)
        grads["db" + str(l+1)] = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
        grads["dA" + str(l+1)] = np.dot(linear_cache[1].T, dZ)


    return grads











