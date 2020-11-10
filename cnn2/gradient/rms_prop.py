import numpy as np
from gradient.abs_gradient import *

class RMSprop(ABSGradient):

    def __init__(self, lr, beta, exp):
        super(RMSprop, self).__init__(lr)

        self.beta = beta

        self.exp = exp

        self.delta_weight = None
        self.delta_bias = None

        self.rms_weight = None
        self.rms_bias = None

        self.size = 0

    def setShape(self, weightShape, biasShape):

        self.delta_weight = np.zeros(weightShape)
        self.delta_bias = np.zeros(biasShape)

        self.rms_weight = np.zeros(weightShape)
        self.rms_bias = np.zeros(biasShape)

        self.size = 0

    def put(self, grain_weight, grain_bias):

        self.delta_weight += grain_weight
        self.delta_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        self.rms_weight = self.beta * self.rms_weight + (1 - self.beta) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.rms_weight + self.exp))

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        self.rms_bias = self.beta * self.rms_bias + (1 - self.beta) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.rms_bias +  self.exp))

    def reset(self):

        self.delta_weight = np.zeros(self.delta_weight.shape)
        self.delta_bias = np.zeros(self.delta_bias.shape)
        self.size = 0
