from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *
from model import *


gradient = Adam(lr=0.001, beta1=0.95, beta2=0.95)

train_x, train_y = extractMNIST('./mnist/train')
train_x -= int(np.mean(train_x))
train_x /= int(np.std(train_x))

test_x, test_y = extractMNIST('./mnist/test')
test_x -= int(np.mean(test_x))
test_x /= int(np.std(test_x))

layerInfoList = [
    {'type':'input', 'parameter':{'input_shape':train_x.shape[1:]}},
    {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':False, 'gradient':gradient.copy()}},
    {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':False, 'gradient':gradient.copy()}},
    {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
    {'type':'flatten', 'parameter':{}},
    {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient.copy()}},
    {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient.copy()}},
    {'type':'dense', 'parameter':{'units':10, 'activation':'softmax', 'gradient':gradient.copy()}}]


model = Model(layerInfoList)
model.build()
model.train(train_x, train_y, epochs=1)
prediction = model.predict(test_x)

count = len(prediction)
correct = 0

for i in range(count):
    if np.argmax(prediction[i]) == test_y[i]:
        correct += 1
        print('ok', test_y[i])
    else:
        print('false', test_y[i])


print('accuracy : ', float(correct / count) * 100, '%')
