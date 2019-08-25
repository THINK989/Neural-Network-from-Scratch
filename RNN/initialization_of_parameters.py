import numpy as np


np.random.seed(9) #used so that we get the same random values generated everytime
"""
    Layer dimension is an array with values specifying number of nodes in each layer
    eg:- [3,4,2]
    here,
        layer1 has 3 nodes
                            #weights1 with shape(4,3) and #bias1 with shape(4,1)
        layer2 has 4 nodes 
                            #weights2 with shape(2,4) and #bias2 with shape(2,1) 
        layer3 has 2 nodes
"""

"""
Functions used:-
np.zeroes(){visit}:- https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
np.random.randn(){visit}:- https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.rand.html
np.random.seed(){visit}:- https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
"""


def zero_init(layer_dims):
    parameters = {} #Data dictionary to store the 'key' as weight and bias 
    #and 'value' as array of values defined below 
    L = len(layer_dims)
    for i in range(1,L):
        parameters['W' + str(i)] = np.zeroes((layer_dims[l], layer_dims[l-1]))
        parameters['B' + str(i)] = np.zeroes((layer_dims[l],1))
        
    return parameters

def random_init(layers_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1,L):
        parameters['W' + str(i)] = np.random.randn((layer_dims[l], layer_dims[l-1]))
        parameters['B' + str(i)] = np.zeroes((layer_dims[l],1))
        
    return parameters

def he_init(layers_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1,L):
        parameters['W' + str(i)] = np.random.randn((layer_dims[l], layer_dims[l-1])) * (np.sqrt(2/layers_dim(l-1)))
        parameters['B' + str(i)] = np.zeroes((layer_dims[l],1))
        
    return parameters

def xavier_init(layers_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1,L):
        parameters['W' + str(i)] = np.random.randn((layer_dims[l], layer_dims[l-1])) * (np.sqrt(1/layers_dim(l-1)))
        parameters['B' + str(i)] = np.zeroes((layer_dims[l],1))
        
    return parameters










