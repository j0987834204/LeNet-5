# -*- coding: utf-8 -*-
import numpy as np


class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def _forward(self, X):
        #print("ReLU: _forward")
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX

class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        # Prevent overflow.
        X = np.clip(X, -500, 500)
        self.cache = X
        return 1 / (1 + np.exp(-X))

    def _backward(self, dout):
        X = self.cache
        dX = dout*X*(1-X)
        return dX

class tanh():
    """
    tanh activation layer
    """
    def __init__(self):
        self.cache = X

    def _forward(self, X):
        self.cache = X
        return np.tanh(X)

    def _backward(self, X):
        X = self.cache
        dX = dout*(1 - np.tanh(X)**2)
        return dX

class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        #print("Softmax: _forward")
        # Prevent overflow.
        X = np.clip(X, -500, 500)
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.l = len(params)
        self.parameters = params
        self.moumentum = []
        self.velocities = []
        self.m_cat = []
        self.v_cat = []
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
            self.moumentum.append(np.zeros(param['val'].shape))
            self.v_cat.append(np.zeros(param['val'].shape))
            self.m_cat.append(np.zeros(param['val'].shape))

    
    def step(self):
        self.t += 1
        for i in range(self.l):
            g = self.parameters[i]['grad']
            self.moumentum[i]  = self.beta1 * self.moumentum[i]  + (1 - self.beta1) * g
            self.velocities[i] = self.beta2 * self.velocities[i] + (1 - self.beta2) * g * g
            self.m_cat[i] = self.moumentum[i]  / (1 - self.beta1 ** self.t)
            self.v_cat[i] = self.velocities[i] / (1 - self.beta2 ** self.t) 
            self.parameters[i]['val'] -= self.lr * self.m_cat[i] / (self.v_cat[i] ** 0.5 + self.epislon)