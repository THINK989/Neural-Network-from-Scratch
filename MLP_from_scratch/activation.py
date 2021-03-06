import numpy as np

# THIS FILE CONTAINS A SCRATCH IMPLEMENTATION FOR MOST COMMONLY USED ACTIVATION FUNCTIONS
"""
    Compute the activation(whichever activation you wants to use) of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- activation(x)
"""
"""
Function Used
np.exp(){visit}:- https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
np.tanh(){visit}:- https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html

"""

def sigmoid(x):
    #Normally used to find the probability i.e. used in output layer
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    # Most widely used and is used in hidden layers
    # it resticts the value to be greater than or equal to zero
    s = np.maximum(0,x)
    
    return s
"""
A RELU Activation can sometimes cause a neuron to die in other words, 
causes a neuron to never activate again.
"""
def Leaky_RELU(x):
    #Leaky RELU activation helps in aiding the dying problem of relu,increasing the threshold for x.
    #This causes x to be significantly more than 0.
    s = max(x*0.01, x)
    return s


def Tanh(x):
    # It squashes x to the range [-1, 1]. Its output is Zero Centered.
    s = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    #         or
    #s = np.tanh(x)
    return s


def HardTanh(x):
    #while tanh might output a value more than 1 or less than -1.
    #HardTanh Activation forces the value to be in range[-1,1].
    if x > 1:
        x = 1
    elif x < -1:
        x = -1
    else:
        x = x
    return x


def ELU(x):
    # A good alternative to relu, cost convergence to zero is faster.
    # ELU can produce negative output
    alpha = 1.0
    s = max(0,x) + min(0, alpha *(np.exp(x) - 1))
    return s


def Softmax(x):
    #Softmax normalizes input into a probability distribution consisting of K probabilities.
    # K probabilities sums up to 1
    s = np.exp(x)/np.sum(np.exp(x))
    return s















    
