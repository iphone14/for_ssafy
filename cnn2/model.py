from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *


class Model:
    def __init__(self, layerList, log='info'):
        self.layerList = layerList
        self.head = None
        self.tail = None
        self.log = log
        self.labelIndexs = None

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
                className = chain.__class__.__name__
                print('name={0:15} output={1}'.format(className, str(chain.outputShape())))

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

        if self.log == 'info':
            print('---------------------Train---------------------')

        classes = y.shape[1]

        for epoch in range(epochs):
            loss = self.batchTrain(self.head, self.tail, x, y, classes)

            if self.log == 'info':
                print('epochs={0:13} loss={1}'.format((str(epoch) +'/' + str(epochs)), str(loss)))

    def encodeOnehot(self, label, classes):
        return np.eye(classes)[label].reshape(classes, 1)

    def categoricalCrossEntropy(self, predict, label):
        return -np.sum(label * np.log2(predict))

    def forwardSoftmax(self, input):
        input = np.exp(input)
        return input / np.sum(input)

    def backwardSoftmax(self, predict, label):
        return predict - label

    def batchTrain(self, head, tail, x, y, classes):

        batches = len(x)

        loss = 0

        for i in range(batches):
            predict = self.forward(head, x[i])

            predict = self.forwardSoftmax(predict)

            label = y[i]

            error = self.backwardSoftmax(predict, label)

            loss += self.categoricalCrossEntropy(predict, label)

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


    def test(self, x, y):

        if self.log == 'info':
            print('---------------------Test---------------------')

        prediction = self.predict(x)

        count = len(prediction)
        correct = 0

        for i in range(count):
            p_index = np.argmax(prediction[i])
            y_index = np.argmax(y[i])
            if p_index == y_index:
                correct += 1
                if self.log == 'info':
                    print('predict={0:12} correct={1}'.format((str(y_index) +'/' + str(p_index)), 'O'))
            else:
                if self.log == 'info':
                    print('predict={0:12} correct={1}'.format((str(y_index) +'/' + str(p_index)), 'X'))

        accuracy = float(correct / count) * 100

        return accuracy
