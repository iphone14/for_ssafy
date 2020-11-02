from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *


gradient = Adam(lr=0.001, beta1=0.95, beta2=0.95)

x, y = extractMNIST('./mnist/train')
x-= int(np.mean(x))
x/= int(np.std(x))


def forward(head, output):

    chain = head

    while True:

        output = chain.forward(output)

        next = chain.forwardChain()

        if next is None:
            break

        chain = next

    return output


def backward(tail, error):

    chain = tail

    last_error = None

    while True:
        error = chain.backward(error)

        last_error = error

        next = chain.backwardChain()

        if next is None:
            break

        chain = next


def updateGradient(head):

    chain = head

    while True:

        chain.updateGradient()

        next = chain.forwardChain()

        if next is None:
            break

        chain = next


I1 = Input(input_shape=x.shape[1:])

C1 = Convolution(filters=3, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=I1, gradient=gradient.copy())
C1_shape = C1.outputShape()
print(C1_shape)


C2 = Convolution(filters=3, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=C1, gradient=gradient.copy())
C2_shape = C2.outputShape()
print(C2_shape)


MP1 = MaxPooling(pool_size=(2, 2), strides=None, chain=C2)
MP1_shape = MP1.outputShape()
print(MP1_shape)

F1 = Flatten(chain=MP1)
F1_shape = F1.outputShape()


D1 = Dense(units=128, activation='linear', chain=F1, gradient=gradient.copy())
D1_shape = D1.outputShape()
print(D1_shape)


D2 = Dense(units=64, activation='linear', chain=D1, gradient=gradient.copy())
D2_shape = D2.outputShape()
print(D2_shape)


D3 = Dense(units=10, activation='softmax', chain=D2, gradient=gradient.copy())
D3_shape = D3.outputShape()
print(D3_shape)



def batchTrain(head, tail, x, y):

    batches = len(x)

    classes = 10

    loss = 0

    for i in range(batches):

        output = forward(head, x[i])

        #print('----forward')

        onehot = labelToOnehot(y[i], classes)

        loss += categoricalCrossEntropy(output, onehot)

        error = output - onehot

        #print('----backward')

        backward(tail, error)

        #print(i)

    print('loss : ', loss / batches)

    updateGradient(head)


epochs = 50

for epoch in range(epochs):
    batchTrain(I1, D3, x, y)
