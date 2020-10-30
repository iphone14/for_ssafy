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

    def copy(self):
        return Adam(self.lr, self.beta1, self.beta2)

    def put(self, grain_weight, grain_bias):
        return None

    def deltaWeight(self):
        return None

    def deltaBias(self):
        return None

    def reset(self):
        return None
