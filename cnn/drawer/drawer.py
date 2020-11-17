from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime
import os

def matrixToImage(matrix):

	step = 255 / (np.max(matrix) - np.min(matrix))

	(rows, height, width) = matrix.shape

	rectSize = 30

	image = Image.new('RGB', (width * rectSize, rows * height * rectSize))
	drawer = ImageDraw.Draw(image)

	for r in range(rows):
		for y in range(height):
			for x in range(width):
				point = (x * rectSize), ((rectSize * r) + y)
				row = matrix[r]
				value = row[y, x]
				color = int(255 - int(value * step))
				color = color if color >= 0 else 0

				drawRect(drawer, point, rectSize, color)
				drawText(drawer, point, rectSize, str(round(value, 1)))

	saveFile(image)


def drawRect(drawer, point, rectSize, color):

	colorHex = '#%02x%02x%02x' % (color, color, color)
	drawer.rectangle([(point[0], point[1]), (point[0] + rectSize, point[1] + rectSize)], colorHex)


def drawText(drawer, point, rectSize, text):

	fontSize = int(rectSize / 3)
	padding = int((rectSize - fontSize) / 2)
	font = ImageFont.truetype("arial.ttf", fontSize)
	colorHex = '#%02x%02x%02x' % (255, 0, 255)
	drawer.text((point[0] + padding - 4, point[1] + padding), text, font=font, fill=colorHex)


def saveFile(image):

	folder = 'images'

	now = datetime.datetime.now()
	strDate = now.strftime("%H%M%S")

	if os.path.isdir(folder) is False:
		os.mkdir(folder)

	image.save(folder + '/' + strDate + '.png')
