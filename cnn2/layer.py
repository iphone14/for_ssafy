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
        return self.backward_chain

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, error):
        pass

    @abstractmethod
    def outputShape(self):
        pass

    @abstractmethod
    def updateGradient(self):
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

    def updateGradient(self):
        pass



class Convolution(Layer):

    def __init__(self, filters, kernel_size, strides, padding, chain, gradient):
        super(Convolution, self).__init__(chain)

        self.strides = strides
        self.padding_size = self.paddingSize(kernel_size[0], kernel_size[1]) if padding else (0,0)

        self.weight = self.initWeight((filters, self.input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((filters, 1))

        self.gradient = gradient

    def initWeight(self, size, scale = 1.0):

        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def appendPadding(self, input):

        size = self.padding_size

        rows = []
        for i in input:
            rows.append(np.pad(i, ((size[0], size[0]),(size[1], size[1])), 'constant', constant_values=0))

        return np.array(rows)

    def forward(self, input):

        input = self.appendPadding(input)

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (input_colors, input_height, input_width) = input.shape
        (stride_y, stride_x) = self.strides

        assert colors == input_colors, "filter miss match"

        output = np.zeros(self.outputShape())

        for filter in range(filters):
            y = out_y = 0
            while (y + kernel_height) <= input_height:
                x = out_x = 0
                while (x + kernel_width) <= input_width:
                    output[filter, out_y, out_x] = np.sum(self.weight[filter] * input[:, y:y + kernel_height, x:x + kernel_width]) + self.bias[filter]
                    x += stride_x
                    out_x += 1

                y += stride_y
                out_y += 1

        output[output<=0] = 0

        return output

    def backward(self, input):
        print('back conv')
        return None

    def paddingSize(self, kernel_height, kernel_width):

        return ((kernel_height - 1) // 2, (kernel_width - 1) // 2)

    def outputShape(self):

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (stride_y, stride_x) = self.strides

        numerator_height = ((self.padding_size[0] * 2) - kernel_height) + self.input_shape[1]
        numerator_width = ((self.padding_size[1] * 2) - kernel_width) + self.input_shape[2]

        calc_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)

        return (filters,) + calc_shape


    def updateGradient(self):
        pass


class MaxPooling(Layer):

    def __init__(self, pool_size, strides, chain):
        super(MaxPooling, self).__init__(chain)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides

    def forward(self, input):

        (input_colors, input_height, input_width) = input.shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        height = int(input_height // stride_y)
        width = int(input_width // stride_x)

        output = np.zeros((input_colors, height, width))

        for color in range(input_colors):
            y = out_y = 0
            while (y + pool_height) <= input_height:
                x = out_x = 0
                while (x + pool_width) <= input_width:
                    output[color, out_y, out_x] = np.max(input[color, y:y + stride_y, x:x + stride_x])
                    x += stride_x
                    out_x += 1
                y += stride_y
                out_y += 1

        return output

    def backward(self, error):
        print('back max')
        return None

    def outputShape(self):

        calc_shape = ((self.input_shape[1] // self.strides[0]), (self.input_shape[2] // self.strides[1]))

        return (self.input_shape[0],) + calc_shape


    def updateGradient(self):
        pass


class Flatten(Layer):

    def __init__(self, chain):
        super(Flatten, self).__init__(chain)

    def forward(self, input):
        return input.reshape((-1, 1))

    def backward(self, error):
        print('back flatten')
        return None

    def outputShape(self):
        return (reduce(operator.mul, self.input_shape), )

    def updateGradient(self):
        pass


class Dense(Layer):

    def __init__(self, units, activation, chain, gradient):
        super(Dense, self).__init__(chain)

        self.units = units
        self.activation = activation

        self.wieght = self.initWeight((units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

        self.input = None

        self.gradient = gradient

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def forward(self, input):

        self.input = input

        output =  self.wieght.dot(input) + self.bias

        if self.activation == 'softmax':
            output = np.exp(output)
            return output/np.sum(output)
        elif self.activation == 'relu':
            output[output<=0] = 0
            return output
        else:   # linear
            return output

    def backward(self, error):

        grain_weight = error.dot(self.input.T)
        grain_bias = np.sum(error, axis = 1).reshape(self.bias.shape)

        self.gradient.put(grain_weight, grain_bias)

        backward_chain_error = self.wieght.T.dot(error)
        #backward_chain_error[self.input <= 0] = 0

        return backward_chain_error

    def outputShape(self):
        return (self.units, )

    def updateGradient(self):
        self.weight += self.gradient.deltaWeight()
        self.bias += self.gradient.deltaBias()
        self.gradient.reset()
