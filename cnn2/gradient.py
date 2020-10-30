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
    def copy(self):
        pass

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

    def __init__(self, lr, beta1, beta2):
        super(Adam, self).__init__(lr)

        self.beta1 = beta1
        self.beta2 = beta2

        self.delta_weight = None
        self.delta_bias = None

        self.adam_weight = None
        self.adam_bias = None

        self.size = 0


    def copy(self):
        return Adam(self.lr, self.beta1, self.beta2)

    def put(self, grain_weight, grain_bias):

        if self.delta_weight == None:
            self.delta_weight = np.zeros(grain_weight.shape)

        if self.delta_bias == None:
            self.delta_bias = np.zeros(grain_bias.shape)

        if self.adam_weight == None:
            self.adam_weight = np.zeros(grain_weight.shape)

        if self.adam_bias == None:
            self.adam_bias = np.zeros(grain_bias.shape)

        self.delta_weight += grain_weight
        self.grain_bias += grain_bias

        self.size += 1

    def deltaWeight(self):

        avg_delta = self.delta_weight / self.size

        self.adam_weight = self.beta1 * self.adam_weight + (1 - self.beta2) * (avg_delta)**2

        return lr * (avg_delta)/(np.sqrt(self.adam_weight) + 1e-7)

    def deltaBias(self):

        avg_delta = self.delta_bias / self.size

        self.adam_bias = self.beta1 * self.adam_bias + (1 - self.beta2) * (avg_delta)**2
        
        return lr * (avg_delta)/(np.sqrt(self.adam_bias) + 1e-7)

    def reset(self):
        self.delta_weight = None
        self.delta_bias = None
        self.size = 0
