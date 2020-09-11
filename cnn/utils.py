'''
Description: Utility methods for a Convolutional Neural Network

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from forward import *
import numpy as np
from PIL import Image
import os

from array import *
from random import shuffle

def getFileList(path):

	fileList = []

	for dirname in os.listdir(path):

		subPath = os.path.join(path, dirname)

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png"):
				fileList.append({"label":dirname, "name":fileName})

	shuffle(fileList)

	return fileList


def extractMNIST(path):

	fileList = getFileList(path)

	X = []
	Y = []

	for fileInfo in fileList:

		label = fileInfo['label']
		name = fileInfo['name']

		fileName = path + '/' + label + '/' + name
		img = Image.open(fileName)

		img = np.array(img).astype(np.float32).flatten()

		X.append(img)
		Y.append(int(label))

	return np.array(X), np.array(Y)



def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
    '''
    Make predictions with trained filters/weights.
    '''
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation

    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity

    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)
