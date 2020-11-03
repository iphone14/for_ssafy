from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *


def className(instance):
    return instance.__class__.__name__


class Model:
    def __init__(self, layerList, log='info'):
        self.layerList = layerList
        self.head = None
        self.tail = None
        self.log = log

    def createModel(self, layerList):

        chain = None
        head = None
        tail = None

        if self.log == 'info':
            print('---------------------Model---------------------')

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

            if self.log == 'info':
                print('name={0:15} output={1}'.format(className(chain),str(chain.outputShape())))

            if head == None:
                head = chain

        tail = chain

        if self.log == 'info':
            print('------------------------------------------------')

        return head, tail

    def build(self):

        head, tail = self.createModel(self.layerList)

        self.head = head
        self.tail = tail

        return head, tail

    def train(self, x, y, epochs):

        for epoch in range(epochs):
            loss = self.batchTrain(self.head, self.tail, x, y)

            if self.log == 'info':
                print('elapse={0:13} loss={1}'.format((str(epoch) +'/' + str(epochs)), str(loss)))

    def labelToOnehot(self, label, classes):
        return np.eye(classes)[label].reshape(classes, 1)

    def categoricalCrossEntropy(self, predict_y, label):
        return -np.sum(label * np.log2(predict_y))


    def batchTrain(self, head, tail, x, y):

        batches = len(x)

        classes = 10

        loss = 0

        for i in range(batches):
            output = self.forward(head, x[i])

            onehot = self.labelToOnehot(y[i], classes)

            loss += self.categoricalCrossEntropy(output, onehot)

            error = output - onehot

            self.backward(tail, error)

        self.updateGradient(head)

        return loss / batches


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
