import numpy as np
import math

#contains code for 1 layer neural network forward pass

def relu(x):
        return np.maximum(np.zeros(x.shape[0]),x)

def linear(x):
        w = np.random.rand(x.shape[1], 1)
        return np.dot(x,w)

def mse(ypred,y):
        return sum((ypred-y)**2)/y.shape[0]


#forward pass
x = np.random.rand(10,3)
y = np.array([0,1,1,0,0,0,1,0,0,1])

z =  linear(x)

act = sigmoid(z)

y_pred = relu(z)

#mean squared error to evaluate loss
loss = mse(y_pred, y)

#backward pass
'''
calculating derivative of weight with respect to total error/loss
d(a/b) - derivative of a wrt to b
As per chain rule :
d_weight = d(error/output) * d(output/activation) * d(activation * weight)
'''
d_output = y_pred - y #derivative of loss wrt y_pred
d_act = y_pred(1 - y_pred) #derivative of output wrt output
d_weight = (y_pred - y)*y_pred(1 - y_pred)*x 

updated_weight = w - d_weight

# in the next forward pass, updated_weights will be used in forward pass



