import numpy as np
from PIL import Image
import os
from array import *
from random import shuffle
import operator


def getFileList(path):

	fileList = []

	for dirname in os.listdir(path):

		subPath = os.path.join(path, dirname)

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png"):
				fileList.append({"label":dirname, "name":fileName})

	#shuffle(fileList)

	fileList = [{'label': '0', 'name': '1.png'}, {'label': '1', 'name': '1.png'}, {'label': '6', 'name': '32.png'}, {'label': '8', 'name': '137.png'}, {'label': '9', 'name': '4.png'}]

	#fileList = [{'label': '0', 'name': '1.png'}]

	return fileList


def extractMNIST(path):

	fileList = getFileList(path)

	colorDim = 1

	X = []
	Y = []

	imgSize = 0

	for fileInfo in fileList:

		label = fileInfo['label']
		name = fileInfo['name']

		filePath = path + '/' + label + '/' + name

		img = np.array(Image.open(filePath)).astype(np.float32)

		imgSize = img.shape[0]

		list = []

		for i in range(colorDim):
			list = np.append(list, img.copy())

		X.append(list)

		Y.append(int(label))

	return np.array(X).reshape(len(X), colorDim, imgSize, imgSize), np.array(Y)


def labelToOnehot(label, classes):
    return np.eye(classes)[label].reshape(classes, 1)

def categoricalCrossEntropy(predict, label):
    return -np.sum(label * np.log2(predict))

def load(name):

	np_path = 'matrix_input' + '/' + name + '.npy'
	return np.load(np_path)

def save(matrix, name):

	text_path = 'matrix_output' + '/' + name + '.txt'
	np.savetxt(text_path, matrix.flatten())
