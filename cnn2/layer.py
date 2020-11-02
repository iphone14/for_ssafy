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
        self.gradient.setShape(self.weight.shape, self.bias.shape)

        self.last_output = None
        self.last_input = None

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

        self.last_input = input

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (input_colors, input_height, input_width) = self.input_shape
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

        self.last_output = output

        return output

    def backward(self, error):

        error[self.last_output<=0] = 0

        (filters, colors, kernel_height, kernel_width) = self.weight.shape
        (input_colors, input_height, input_width) = self.input_shape
        (stride_y, stride_x) = self.strides

        output = np.zeros(self.input_shape)
        grain_weight = np.zeros(self.weight.shape)
        grain_bias = np.zeros(self.bias.shape)

        for filter in range(filters):
            y = out_y = 0
            while (y + kernel_height) <= input_height:
                x = out_x = 0
                while (x + kernel_width) <= input_width:
                    grain_weight[filter] += error[filter, out_y, out_x] * self.last_input[:, y:y + kernel_height, x:x + kernel_width]
                    output[:, y:y + kernel_height, x:x + kernel_width] += error[filter, out_y, out_x] * self.weight[filter]
                    x += stride_x
                    out_x += 1
                y += stride_y
                out_y += 1

            grain_bias[filter] = np.sum(error[filter])

        self.gradient.put(grain_weight, grain_bias)

        return output

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

        deltaWeight = self.gradient.deltaWeight()
        deltaBias = self.gradient.deltaBias()

        self.weight -= deltaWeight
        self.bias -= deltaBias

        self.gradient.reset()


class MaxPooling(Layer):

    def __init__(self, pool_size, strides, chain):
        super(MaxPooling, self).__init__(chain)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides
        self.last_input = None

    def forward(self, input):

        self.last_input = input

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
                    output[color, out_y, out_x] = np.max(input[color, y:y + pool_height, x:x + pool_width])
                    x += stride_x
                    out_x += 1
                y += stride_y
                out_y += 1

        return output

    def nanargmax(self, array):
        idx = np.nanargmax(array)
        return np.unravel_index(idx, array.shape)

    def backward(self, error):

        (input_colors, input_height, input_width) = self.input_shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        output = np.zeros(self.input_shape)

        for color in range(input_colors):
            y = out_y = 0
            while (y + pool_height) <= input_height:
                x = out_x = 0
                while (x + pool_width) <= input_width:
                    (a, b) = self.nanargmax(self.last_input[color, y:y + pool_height, x:x + pool_width])
                    output[color, y + a, x + b] = error[color, out_y, out_x]
                    x += stride_x
                    out_x += 1
                y += stride_y
                out_y += 1

        return output

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
        return error.reshape(self.input_shape)

    def outputShape(self):
        return (reduce(operator.mul, self.input_shape), )

    def updateGradient(self):
        pass


class Dense(Layer):

    def __init__(self, units, activation, chain, gradient):
        super(Dense, self).__init__(chain)

        self.units = units
        self.activation = activation

        self.weight = self.initWeight((units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

        self.last_input = None
        self.last_output = None

        self.gradient = gradient
        self.gradient.setShape(self.weight.shape, self.bias.shape)

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def forward(self, input):

        self.last_input = input

        output = self.weight.dot(input) + self.bias

        if self.activation == 'softmax':
            output = np.exp(output)
            output = output/np.sum(output)
        elif self.activation == 'relu':
            output[output<=0] = 0
        #else:   # linear
            #self.last_output = output

        self.last_output = output

        return output

    def backward(self, error):

        if self.activation == 'linear':
            error[self.last_output <= 0] = 0

        grain_weight = error.dot(self.last_input.T)
        grain_bias = np.sum(error, axis = 1).reshape(self.bias.shape)

        self.gradient.put(grain_weight, grain_bias)

        backward_chain_error = self.weight.T.dot(error)

        return backward_chain_error

    def outputShape(self):
        return (self.units, )

    def updateGradient(self):

        deltaWeight = self.gradient.deltaWeight()
        detalBias = self.gradient.deltaBias()

        self.weight -= deltaWeight
        self.bias -= detalBias

        self.gradient.reset()
