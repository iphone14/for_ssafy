from abc import *
import numpy as np

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

        self.strides = strides
        self.padding = padding
        self.filters = filters
        self.kernel_size = kernel_size

        self.wieght = 1
        self.bias = 1

    def Forward(self):

        conv1 = 1
        return conv1

    def Backward(self):
        print('gege')

    def OutputShape(self):

        padding_size = ((np.array(self.kernel_size) - 1) / 2) if self.padding else np.array((0, 0))

        assert np.all(np.mod(padding_size, 1) == 0.0), "padding_size error"

        calc_shape = ((self.input_shape[-2:] + (padding_size*2) - self.kernel_size) / self.strides) + 1

        output_shape = np.hstack([self.input_shape[0], calc_shape])

        assert np.all(np.mod(output_shape, 1) == 0.0), "output_shape error"

        return output_shape


C1 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, input_shape=(3, 28, 28))
C1.Forward()
C1_shape = C1.OutputShape()
print(C1_shape)

C2 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, input_shape=C1_shape)
print(C2.OutputShape())
