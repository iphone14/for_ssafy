from abc import *
import numpy as np
import operator
from functools import reduce
from utils import *
from gradient import *
from layer import *
from model import *


gradient = Adam(lr=0.001, beta1=0.95, beta2=0.95)

x, y = extractMNIST('./mnist/train')
x-= int(np.mean(x))
x/= int(np.std(x))

layerInfoList = [
    {'type':'input', 'parameter':{'input_shape':x.shape[1:]}},
    {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':False, 'gradient':gradient.copy()}},
    {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':False, 'gradient':gradient.copy()}},
    {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
    {'type':'flatten', 'parameter':{}},
    {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient.copy()}},
    {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient.copy()}},
    {'type':'dense', 'parameter':{'units':10, 'activation':'softmax', 'gradient':gradient.copy()}}]


model = Model(layerInfoList)
head, tail = model.build()
model.train(x, y, epochs=50)
