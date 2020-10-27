from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *


class HiddenLayer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, chain):

        self.input_shape = None
        self.backwardChain = chain
        self.forwardChain = None

        if chain is not None:
            self.input_shape = chain.OutputShape()
            chain.forwardChain = self

    @abstractmethod
    def PassForward(self):
        pass

    @abstractmethod
    def PassBackward(self):
        pass

    def Forward(self):
        self.PassForward()
        return self.forwardChain

    def Backward(self):
        self.PassBackward()
        return self.backwardChain

    @abstractmethod
    def OutputShape(self):
        pass



class Input(HiddenLayer):

    def __init__(self):
        super(Input, self).__init__(chain=None)

    def setShape(self, shape):
        self.input_shape = shape

    def PassForward(self):
        print('input')

    def PassBackward(self):
        print('back input')

    def OutputShape(self):
        return self.input_shape


class Convolution(HiddenLayer):

    def __init__(self, filters, kernel_size, strides, padding, chain):
        super(Convolution, self).__init__(chain)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.weight = self.initWeight((filters, self.input_shape[0], kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((filters, 1))

    def initWeight(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def PassForward(self):
        print('conv')

    def PassBackward(self):
        print('back conv')

    def paddingSize(self):
        return tupleFloorDivDiv(tupleSub(self.kernel_size, (1, 1)), (2, 2)) if self.padding else (0, 0)

    def OutputShape(self):

        padding_size = self.paddingSize()

        numerator = tupleAdd(tupleSub(tupleMul(padding_size, (2, 2)), self.kernel_size), self.input_shape[-2:])

        calc_shape = tupleAdd(tupleFloorDiv(numerator, self.strides), (1, 1))

        output_shape = (self.filters,) + calc_shape

        return output_shape


class MaxPooling(HiddenLayer):

    def __init__(self, pool_size, strides, chain):
        super(MaxPooling, self).__init__(chain)

        self.pool_size = pool_size
        self.strides = strides

    def PassForward(self):
        print('pool')

    def PassBackward(self):
        print('back pool')

    def OutputShape(self):

        cal_strides = self.pool_size if self.strides == None else self.strides

        calc_shape = tupleFloorDiv(self.input_shape[-2:], cal_strides)

        output_shape = (self.input_shape[0],) + calc_shape

        return output_shape


class Flatten(HiddenLayer):

    def __init__(self, chain):
        super(Flatten, self).__init__(chain)

    def PassForward(self):
        print('flatten')

    def PassBackward(self):
        print('back flatten')

    def OutputShape(self):
        return (reduce(operator.mul, self.input_shape), )


class Dense(HiddenLayer):

    def __init__(self, units, chain):
        super(Dense, self).__init__(chain)

        self.units = units

        self.wieght = self.initWeight((units, self.input_shape[0]))
        self.bias = np.zeros((units, 1))

    def initWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def PassForward(self):
        print('dense')

    def PassBackward(self):
        print('back dense')

    def OutputShape(self):
        return (self.units, )


x, y = extractMNIST('./mnist/train')
x-= int(np.mean(x))
x/= int(np.std(x))


I1 = Input()
I1.setShape(x.shape[1:])

C1 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=I1)
C1_shape = C1.OutputShape()
print(C1_shape)

C2 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=C1)
C2_shape = C2.OutputShape()
print(C2_shape)

MP1 = MaxPooling(pool_size=(2, 2), strides=None, chain=C2)
MP1_shape = MP1.OutputShape()
print(MP1_shape)

F1 = Flatten(chain=MP1)
F1_shape = F1.OutputShape()
print(F1_shape)


D1 = Dense(units=128, chain=F1)
D1_shape = D1.OutputShape()
print(D1_shape)


D2 = Dense(units=64, chain=D1)
D2_shape = D2.OutputShape()
print(D2_shape)


D3 = Dense(units=10, chain=D2)
D3_shape = D3.OutputShape()
print(D3_shape)


chain = I1


for i in range(10):

    while True:
        next = chain.Forward()

        if next is None:
            break
        chain = next


    while True:
        next = chain.Backward()

        if next is None:
            break
        chain = next
