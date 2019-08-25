import numpy as np



def GD(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters['W' + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l+1)] - learning_rate*grads['db' + str(l+1)]
    return parameters




def Momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
       
        v["dW" + str(l+1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l+1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)] 

        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
        
    return parameters, v

def Adam(parameters, grads, v, s, t,learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    L = len(parameters) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         
    
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
       
        v_corrected["dW" + str(l+1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2) 
        s["db" + str(l+1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        
        s_corrected["dW" + str(l+1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - (learning_rate * v_corrected["dW" + str(l + 1)]) / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["W" + str(l + 1)] - (learning_rate * v_corrected["dW" + str(l + 1)]) / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        

    return parameters, v, s

def rms_prop(parameters, grads, s, beta, learning_rate, epsilon):
    L = len(parameters) // 2 
    for l in range(L):
       
        s["dW" + str(l+1)] = beta * s["dW" + str(l + 1)] + (1 - beta) * np.power(grads['dW' + str(l + 1)], 2) 
        s["db" + str(l+1)] = beta * s["db" + str(l + 1)] + (1 - beta) * np.power(grads['db' + str(l + 1)], 2)

        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)]) / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)]) / np.sqrt(s["db" + str(l + 1)] + epsilon)
                                                                                     
    return parameters, s












