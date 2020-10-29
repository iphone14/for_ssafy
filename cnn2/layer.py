from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, chain):

        self.input_shape = None
        self.backward_chain = chain
        self.forward_chain = None

        if chain is not None:
            self.input_shape = chain.outputShape()
            chain.forward_chain = self

    def forwardChain(self):
        return self.forward_chain

    def backwardChain(self):
        return self.backward-chain

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, error):
        pass

    @abstractmethod
    def outputShape(self):
        pass


class Input(Layer):

    def __init__(self, input_shape):
        super(Input, self).__init__(chain=None)
        self.input_shape = input_shape

    def forward(self, input):
        print('input')
        return input

    def backward(self, error):
        return error

    def outputShape(self):
        return self.input_shape


class Convolution(Layer):

    def __init__(self, filters, kernel_size, strides, padding, chain, gradient):
        super(Convolution, self).__init__(chain)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.weight = self.initWeight((filters, self.input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((filters, 1))

        self.gradient = gradient

    def initWeight(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def forward(self, input):
        print('conv')
        return None

    def backward(self, input):
        print('back conv')
        return None

    def paddingSize(self):

        if self.padding == True:
            return ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        else:
            return (0,0)

    def outputShape(self):

        padding_size = self.paddingSize()

        numerator_X = ((padding_size[0] * 2) - self.kernel_size[0]) + self.input_shape[1]
        numerator_Y = ((padding_size[1] * 2) - self.kernel_size[1]) + self.input_shape[2]

        calc_shape = ((numerator_X // self.strides[0]) + 1, (numerator_Y // self.strides[1]) + 1)

        output_shape = (self.filters,) + calc_shape

        return output_shape


class MaxPooling(Layer):

    def __init__(self, pool_size, strides, chain):
        super(MaxPooling, self).__init__(chain)

        self.pool_size = pool_size
        self.strides = strides

    def forward(self, input):
        print('max')
        return None

    def backward(self, error):
        print('back max')
        return None

    def outputShape(self):

        cal_strides = self.pool_size if self.strides == None else self.strides

        calc_shape = ((self.input_shape[1] // cal_strides[0]), (self.input_shape[2] // cal_strides[1]))

        output_shape = (self.input_shape[0],) + calc_shape

        return output_shape


class Flatten(Layer):

    def __init__(self, chain):
        super(Flatten, self).__init__(chain)

    def forward(self, input):
        print('flatten')
        return None

    def backward(self, error):
        print('back flatten')
        return None

    def outputShape(self):
        return (reduce(operator.mul, self.input_shape), )


class Dense(Layer):

    def __init__(self, units, chain, gradient):
        super(Dense, self).__init__(chain)

        self.units = units

        self.wieght = self.initWeight((units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

        self.input = None

        self.gradient = gradient

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def forward(self, input):

        self.input = input
        return self.wieght.dot(input) + bias

    def backward(self, error):

        grain_weight = error.dot(self.input.T)
        grain_bias = np.sum(error, axis = 1).reshape(self.bias.shape)

        self.gradient.put(grain_weight, grain_bias)

        if self.gradient.isFull() == True:
            self.weight += self.gradient.deltaWeight()
            self.bias += self.gradient.deltaBias()
            self.gradient.reset()

        backward_chain_error = self.wieght.T.dot(error)
        backward_chain_error[self.self.input <= 0] = 0

        return backward_chain_error

    def outputShape(self):
        return (self.units, )
