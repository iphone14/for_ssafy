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

        showColumn = True

        for layer in layerList:
            parameter = layer['parameter']
            parameter['chain'] = chain

            layerClass = {'input':Input, 'convolution':Convolution, 'maxPooling':MaxPooling, 'flatten':Flatten, 'dense':Dense}
            type = layer['type']

            chain = layerClass[type](**parameter)

            if self.log == 'info':
                table = {'Layer':[chain.__class__.__name__], 'Output Shape':[chain.outputShape()]}
                print_table(table, showColumn)
                showColumn = False

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

        classes = y.shape[1]

        showColumn = True

        for epoch in range(epochs):
            loss = self.batchTrain(self.head, self.tail, x, y, classes)

            if self.log == 'info':
                table = {'Epochs':[str(epoch) +'/' + str(epochs)], 'Loss':[loss]}
                print_table(table, showColumn)
                showColumn = False


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
            predict = self.forward(self.head, x)

            predict = self.forwardSoftmax(predict)
            prediction.append(predict)

        return prediction


    def test(self, x, y):

        prediction = self.predict(x)

        count = len(prediction)
        correct_count = 0

        showColumn = True

        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

        for i in range(count):

            p_index = np.argmax(prediction[i])
            y_index = np.argmax(y[i])

            correct_count += (1 if p_index == y_index else 0)

            if self.log == 'info':
                correct = 'O' if p_index == y_index else 'X'
                y_label = y[i].reshape(-1).round(decimals=2)
                y_predict = prediction[i].reshape(-1).round(decimals=2)
                table = {'Predict':[y_predict], 'Label':[y_label], 'Correct':[correct]}
                print_table(table, showColumn)
                showColumn = False

        accuracy = float(correct_count / count) * 100

        return accuracy
