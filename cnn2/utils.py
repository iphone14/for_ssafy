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

	shuffle(fileList)

	#fileList = [{'label': '0', 'name': '1.png'}, {'label': '1', 'name': '1.png'}, {'label': '6', 'name': '32.png'}, {'label': '8', 'name': '137.png'}, {'label': '9', 'name': '4.png'}]

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


def normalize(x):

	x -= int(np.mean(x))
	x /= int(np.std(x))

	return x


def print_table(table, showColumn):

	template = ''

	for key in table:
		template += '{' + key + ':30}'

	if showColumn == True:
		print('')
		print('='*70)
		colmun = {}

		for key in table:
			colmun[key] = key

		print(template.format(**colmun))

		print('-'*70)

	firstKey = list(table.keys())[0]
	length = len(table[firstKey])

	for i in range(length):
		dict = {}
		for key in table:
			dict[key] = str(table[key][i])

		print(template.format(**dict))
