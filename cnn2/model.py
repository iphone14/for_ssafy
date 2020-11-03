from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *


class Model:
    def __init__(self, layerList):
        self.layerList = layerList
        self.head = None
        self.tail = None

    def createModel(self, layerList):

        chain = None
        head = None
        tail = None

        for layer in layerList:
            type = layer['type']
            parameter = layer['parameter']
            parameter['chain'] = chain

            if type == 'input':
                chain = Input(**parameter)
            elif type == 'convolution':
                chain = Convolution(**parameter)
            elif type == 'maxPooling':
                chain = MaxPooling(**parameter)
            elif type == 'flatten':
                chain = Flatten(**parameter)
            elif type == 'dense':
                chain = Dense(**parameter)

            if head == None:
                head = chain

        tail = chain

        return head, tail

    def build(self):

        head, tail = self.createModel(self.layerList)

        self.head = head
        self.tail = tail

        return head, tail

    def train(self, x, y, epochs):

        for epoch in range(epochs):
            self.batchTrain(self.head, self.tail, x, y)


    def batchTrain(self, head, tail, x, y):

        batches = len(x)

        classes = 10

        loss = 0

        for i in range(batches):
            output = self.forward(head, x[i])

            onehot = labelToOnehot(y[i], classes)

            loss += categoricalCrossEntropy(output, onehot)

            error = output - onehot

            self.backward(tail, error)

        print('loss : ', loss / batches)

        self.updateGradient(head)


    def forward(self, head, output):

        chain = head

        while True:
            output = chain.forward(output)

            next = chain.forwardChain()

            if next is None:
                break

            chain = next

        return output


    def backward(self, tail, error):

        chain = tail

        last_error = None

        while True:
            error = chain.backward(error)

            last_error = error

            next = chain.backwardChain()

            if next is None:
                break

            chain = next


    def updateGradient(self, head):

        chain = head

        while True:
            chain.updateGradient()

            next = chain.forwardChain()

            if next is None:
                break

            chain = next


    def predict(self, test_x):

        prediction = []

        for x in test_x:
            output = self.forward(self.head, x)
            prediction.append(output)

        return prediction
