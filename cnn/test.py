from abc import *

class HiddenLayer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, input_shape):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass



class Convolution(HiddenLayer):

    def __init__(self, input_shape):

        print(input_shape)

        self.wieght = 1
        self.bias = 1

        self.x = 1
        self.stride = 2

    def forward(self):

        conv1 = 1

        self.x = 1

        return conv1

    def backward(self):
        backward('backward')



Convolution = Convolution((3,3))
Convolution.forward()
Convolution.aa
