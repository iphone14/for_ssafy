import numpy as np
from layer.abs_layer import *
from gradient.creator import *

class Dense(ABSLayer):

    def __init__(self, units, activation, backward_layer, gradient):
        super(Dense, self).__init__(backward_layer)

        self.units = units
        self.activation = activation

        self.weight = self.initWeight((units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

        self.last_input = None
        self.last_output = None

        self.gradient = createGradient(gradient)
        self.gradient.setShape(self.weight.shape, self.bias.shape)

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def forward(self, input):

        self.last_input = input

        output = self.weight.dot(input) + self.bias

        if self.activation == 'relu':
            output[output<=0] = 0
        #elif self.activation == 'linear':

        self.last_output = output

        return output

    def backward(self, error):

        if self.activation == 'relu':
            error[self.last_output <= 0] = 0

        grain_weight = error.dot(self.last_input.T)
        grain_bias = np.sum(error, axis = 1).reshape(self.bias.shape)

        self.gradient.put(grain_weight, grain_bias)

        return self.weight.T.dot(error)

    def outputShape(self):
        return (self.units, )

    def updateGradient(self):

        deltaWeight = self.gradient.deltaWeight()
        detalBias = self.gradient.deltaBias()

        self.weight -= deltaWeight
        self.bias -= detalBias

        self.gradient.reset()
