from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *


class Gradient(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def put(self, grain_weight, grain_bias):
        pass

    @abstractmethod
    def deltaWeight(self):
        pass

    @abstractmethod
    def deltaBias(self):
        pass

    @abstractmethod
    def reset(self):
        pass




class Adam(Gradient):

    def __init__(self, lr, beta1, beta2, exp):
        super(Adam, self).__init__(lr)

        self.beta1 = beta1
        self.beta2 = beta2

        self.delta_weight = None
        self.delta_bias = None

        self.adam_weight = None
        self.adam_bias = None

        self.exp = exp

        self.size = 0

    def setShape(self, weightShape, biasShape):

        self.delta_weight = np.zeros(weightShape)
        self.delta_bias = np.zeros(biasShape)

        self.adam_weight = np.zeros(weightShape)
        self.adam_bias = np.zeros(biasShape)

        self.size = 0

    def put(self, grain_weight, grain_bias):

        self.delta_weight += grain_weight
        self.delta_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        self.adam_weight = self.beta1 * self.adam_weight + (1 - self.beta2) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.adam_weight) + self.exp)

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        self.adam_bias = self.beta1 * self.adam_bias + (1 - self.beta2) * (avg_delta)**2

        return self.lr * (avg_delta)/(np.sqrt(self.adam_bias) + self.exp)

    def reset(self):

        self.delta_weight = np.zeros(self.delta_weight.shape)
        self.delta_bias = np.zeros(self.delta_bias.shape)
        self.size = 0




class SGD(Gradient):

    def __init__(self, lr):
        super(SGD, self).__init__(lr)

        self.delta_weight = None
        self.delta_bias = None

        self.size = 0

    def setShape(self, weightShape, biasShape):

        self.delta_weight = np.zeros(weightShape)
        self.delta_bias = np.zeros(biasShape)

        self.size = 0

    def put(self, grain_weight, grain_bias):

        self.delta_weight += grain_weight
        self.delta_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        return self.lr * avg_delta

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        return self.lr * avg_delta

    def reset(self):

        self.delta_weight = np.zeros(self.delta_weight.shape)
        self.delta_bias = np.zeros(self.delta_bias.shape)
        self.size = 0





class RMSprop(Gradient):

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


def createGradient(gradient):

    type = gradient['type']
    parameter = gradient['parameter']

    if type == 'RMSprop':
        return RMSprop(**parameter)
    elif type == 'Adam':
        return Adam(**parameter)
    elif type == 'SGD':
        return SGD(**parameter)

    return None
