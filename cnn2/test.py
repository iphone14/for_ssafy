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

C1 = Convolution(file='if1', name='conv1', filters=3, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=I1, gradient=gradient.copy())
C1_shape = C1.outputShape()
print(C1_shape)


C2 = Convolution(file='if2', name='conv2', filters=3, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=C1, gradient=gradient.copy())
C2_shape = C2.outputShape()
print(C2_shape)


MP1 = MaxPooling(pool_size=(2, 2), strides=None, chain=C2)
MP1_shape = MP1.outputShape()
print(MP1_shape)


F1 = Flatten(chain=MP1)
F1_shape = F1.outputShape()



D1 = Dense(file='iw3', name='dw3', units=128, activation='linear', chain=F1, gradient=gradient.copy())
D1_shape = D1.outputShape()
print(D1_shape)


D2 = Dense(file='iw4', name='dw4', units=64, activation='linear', chain=D1, gradient=gradient.copy())
D2_shape = D2.outputShape()
print(D2_shape)


D3 = Dense(file='iw5', name='dw5', units=10, activation='softmax', chain=D2, gradient=gradient.copy())
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


epochs = 10

for epoch in range(epochs):
    batchTrain(I1, D3, x, y)

"""

output = forward(I1, x[0])

onehot = labelToOnehot(y[0], classes)

loss = categoricalCrossEntropy(output, onehot)

print('loss : ', loss)

error = output - onehot

backward(D3, error)

updateGradient(D3)





print(y[0])
output = x[0]
output = chain.forward(output)
chain = chain.forwardChain()

print('^^^^', output, '^^^^')
output = chain.forward(output)
print('---', output.shape, '------')
print(output)

chain = chain.forwardChain()

print(output)
output = chain.forward(output)
print('---', output.shape, '------')
print(output)

chain = chain.forwardChain()

print(output)
output = chain.forward(output)
print('---', output.shape, '------')
print(output)



chain =  I1

output = x


for i in range(1):

    for m in range(100):

        while True:
            output = chain.forward(output)

            next = chain.forwardChain()

            if next is None:
                break

            chain = next

            #error = output - y
        error = output

        while True:
            error = chain.backward(error)

            next = chain.backwardChain()

            if next is None:
                break
            chain = next
"""
