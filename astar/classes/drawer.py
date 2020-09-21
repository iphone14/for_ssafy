from PIL import Image, ImageDraw, ImageFont
from classes.point import Point
import os

step = 0


def draw(map):

    width = map.width
    height = map.height
    length = 120

    image = Image.new('RGB', (width * length, height * length), color = 'white')

    for y in range(height):
        for x in range(width):
            node = map.getNode(Point(x, y))
            drawNode(node, image, width, height, length)

    saveFile(image)


def saveFile(image):

    global step

    folder = 'process'

    if os.path.isdir(folder) is False:
        os.mkdir(folder)

    image.save(folder + '/' + str(step) + '.png')

    step += 1


def drawCost(node, startX, startY, padding, drawer):

    font = ImageFont.truetype("arial.ttf", 18)
    cost = str(node.costG()) + ' + ' + str(node.costH()) + ' = ' + str(node.costG() + node.costH())

    drawer.text((startX + padding, startY + padding), cost, font=font, fill=(0, 0, 0))


def drawRect(node, startX, startY, length, drawer):

    colorMap = {'c':'green', 'b':'black', 'o':'orange', 'e':'white'}

    drawer.rectangle([(startX, startY), (startX + length, startY + length)], colorMap[node.getState()], 'black',  1)


def drawTriangle(node, startX, startY, length, drawer):

    drawer.polygon([(50, 50), (60, 90), (40, 90)], fill='yellow')

    if node.getParent() is None:
        return

    diff = node.getParent().getPoint() - node.getPoint()

    print(diff)

def drawNode(node, image, width, height, length):

    drawer = ImageDraw.Draw(image)

    point = node.getPoint()
    padding = length * 0.05

    startX = point.x * length
    startY = (height - point.y - 1) * length

    drawRect(node, startX, startY, length, drawer)

    if node.getState() == 'c' or node.getState() == 'o':
        drawCost(node, startX, startY, padding, drawer)
        drawTriangle(node, startX, startY, length, drawer)



        #draw.line((0, 0) + img.size, fill=128)
