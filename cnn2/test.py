from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *

gradient = Adam(lr=0.001, batches=100, beta1=0.95, beta2=0.95)

x, y = extractMNIST('./mnist/train')
x-= int(np.mean(x))
x/= int(np.std(x))


I1 = Input(input_shape=x.shape[1:])

C1 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=I1, gradient=gradient.copy())
C1_shape = C1.outputShape()
print(C1_shape)


C2 = Convolution(filters=5, kernel_size=(3, 3), strides=(1, 1), padding=False, chain=C1, gradient=gradient.copy())
C2_shape = C2.outputShape()
print(C2_shape)


MP1 = MaxPooling(pool_size=(2, 2), strides=None, chain=C2)
MP1_shape = MP1.outputShape()
print(MP1_shape)


F1 = Flatten(chain=MP1)
F1_shape = F1.outputShape()
print(F1_shape)



D1 = Dense(units=128, chain=F1, gradient=gradient.copy())
D1_shape = D1.outputShape()
print(D1_shape)



D2 = Dense(units=64, chain=D1, gradient=gradient.copy())
D2_shape = D2.outputShape()
print(D2_shape)



D3 = Dense(units=10, chain=D2, gradient=gradient.copy())
D3_shape = D3.outputShape()
print(D3_shape)


chain = I1
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

"""



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
