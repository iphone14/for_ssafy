from abc import *

class HiddenLayer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Convolution(HiddenLayer):




    def __init__(self):

        self.wieght
        self.bias

        self.x
        self.stride = 2

    def forward(self):

        conv1 = convolution(x, f1, b1, stride) # convolution operation

        self.x = x

        return conv1

    def backward(self):
        backward('backward')

Convolution = Convolution()
Convolution.forward()




































class ReLU : HiddenLayer

    self.conv

    def forward:
        conv2[conv2<=0] = 0
        self.conv = conv2

        return conv2

    def backward:
        dconv2[self.conv2<=0] = 0

        return dconv2




class Convolution : Network

    stride = 2

    self.f1
    self.b1

    self.x

    def forward(x):

        conv1 = convolution(x, f1, b1, stride) # convolution operation

        self.x = x

        return conv1


    def backward(dconv):

        dimage, df1, db1 = convolutionBackward(dconv, self.x, self.f1, stride) # backpropagate previous gradient through first convolutional layer.

        return dimage

        dconv1, df2, db2 =

        return convolutionBackward(dconv, conv1, f2, stride) # backpropagate previous gradient through second convolutional layer.



class MaxPool : Network

    stride = 2




class Dens : Network







class Output

class SoftMax : Output
