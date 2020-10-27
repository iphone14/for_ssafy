from abc import *
import numpy as np
import operator
from functools import reduce


def tupleAdd(src1, src2):
    return tuple(map(operator.add, src1, src2))

def tupleSub(src1, src2):
    return tuple(map(operator.sub, src1, src2))

def tupleTrueDiv(src1, src2):
    return tuple(map(operator.truediv, src1, src2))

def tupleFloorDiv(src1, src2):
    return tuple(map(operator.floordiv, src1, src2))

def tupleMul(src1, src2):
    return tuple(map(operator.mul, src1, src2))

def tupleMul(src1, src2):
    return tuple(map(operator.mul, src1, src2))




class HiddenLayer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def Forward(self):
        pass

    @abstractmethod
    def Backward(self):
        pass

    @abstractmethod
    def OutputShape(self):
            pass


class Convolution(HiddenLayer):

    def __init__(self, filters, kernel_size, strides, padding, input_shape):
        super(Convolution, self).__init__(input_shape)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.weight = self.initWeight((filters, input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((filters, 1))

    def initWeight(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def Forward(self):
        conv1 = 1
        return conv1

    def Backward(self):
        return None

    def paddingSize(self):
        return tupleFloorDivDiv(tupleSub(self.kernel_size, (1, 1)), (2, 2)) if self.padding else (0, 0)

    def OutputShape(self):

        padding_size = self.paddingSize()

        numerator = tupleAdd(tupleSub(tupleMul(padding_size, (2, 2)), self.kernel_size), self.input_shape[-2:])

        calc_shape = tupleAdd(tupleFloorDiv(numerator, self.strides), (1, 1))

        output_shape = (self.filters,) + calc_shape

        return output_shape


class MaxPooling(HiddenLayer):

    def __init__(self, pool_size, strides, input_shape):
        super(MaxPooling, self).__init__(input_shape)

        self.pool_size = pool_size
        self.strides = strides

    def Forward(self):
        return None

    def Backward(self):
        return None

    def OutputShape(self):

        cal_strides = self.pool_size if self.strides == None else self.strides

        calc_shape = tupleFloorDiv(self.input_shape[-2:], cal_strides)

        output_shape = (self.input_shape[0],) + calc_shape

        return output_shape


class Flatten(HiddenLayer):

    def __init__(self, input_shape):
        super(Flatten, self).__init__(input_shape)

    def Forward(self):
        return None

    def Backward(self):
        return None

    def OutputShape(self):
        return (reduce(operator.mul, self.input_shape), )



class Dense(HiddenLayer):

    def __init__(self, units, input_shape):
        super(Dense, self).__init__(input_shape)

        self.units = units

        self.wieght = self.initWeight((units, input_shape[0]))
        self.bias = np.zeros((units, 1))

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def Forward(self):
        return None

    def Backward(self):
        return None

    def OutputShape(self):
        return (self.units, )


C1 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, input_shape=(1, 28, 28))
C1_shape = C1.OutputShape()
print(C1_shape)

C2 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, input_shape=C1_shape)
C2_shape = C2.OutputShape()
print(C2_shape)

MP1 = MaxPooling(pool_size=(2, 2), strides=None, input_shape=C2_shape)
MP1_shape = MP1.OutputShape()
print(MP1_shape)

F1 = Flatten(input_shape=MP1_shape)
F1_shape = F1.OutputShape()
print(F1_shape, '-faltten')


D1 = Dense(units=128, input_shape=F1_shape)
D1_shape = D1.OutputShape()
print(D1_shape)


D2 = Dense(units=64, input_shape=D1_shape)
D2_shape = D2.OutputShape()
print(D2_shape)


D3 = Dense(units=10, input_shape=D2_shape)
D3_shape = D3.OutputShape()
print(D3_shape)
