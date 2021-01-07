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



x = np.random.rand(10,3)
y = np.array([0,1,1,0,0,0,1,0,0,1])

z =  linear(x)

z = z.squeeze(1)

target = relu(z)

target = target.round()

#mean squared error to evaluate loss
loss = mse(target, y)

print(loss)

