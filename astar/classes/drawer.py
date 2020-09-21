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


def drawCost(node, start, padding, drawer):

    font = ImageFont.truetype("arial.ttf", 18)
    cost = str(node.costG()) + ' + ' + str(node.costH()) + ' = ' + str(node.costG() + node.costH())

    drawer.text((start.x + padding, start.y + padding), cost, font=font, fill=(0, 0, 0))


def drawRect(node, start, length, drawer):

    colorMap = {'c':'green', 'b':'black', 'o':'orange', 'e':'white'}

    drawer.rectangle([(start.x, start.y), (start.x + length, start.y + length)], colorMap[node.getState()], 'black',  1)


def drawTriangle(node, start, length, drawer):

    if node.getParent() is None:
        return

    centerX = start.x + (length / 2)
    centerY =  start.y + (length / 2) + 15

    crossLength = length * 0.25

    angleLength = crossLength * 0.707

    lineStart = None
    lineEnd = None

    diff = node.getPoint() - node.getParent().getPoint()

    diffList = {str(Point(1, 0)):[Point(-crossLength, 0), Point(crossLength, 0)],
                str(Point(-1, 0)):[Point(crossLength, 0), Point(-crossLength, 0)],
                str(Point(0, 1)):[Point(0, crossLength), Point(0, -crossLength)],
                str(Point(0, -1)):[Point(0, -crossLength), Point(0, crossLength)],
                str(Point(1, 1)):[Point(-angleLength, angleLength), Point(angleLength, -angleLength)],
                str(Point(1, -1)):[Point(-angleLength, -angleLength), Point(angleLength, angleLength)],
                str(Point(-1, -1)):[Point(angleLength, -angleLength), Point(-angleLength, angleLength)],
                str(Point(-1, 1)):[Point(angleLength, angleLength), Point(-angleLength, -angleLength)]}

    findDiff = diffList[str(diff)]

    if findDiff is None:
        return

    lineStart = findDiff[0] + Point(centerX, centerY)
    lineEnd = findDiff[1] + Point(centerX, centerY)

    if lineStart is not None and lineEnd is not None:
        radius = 5
        leftUp = (lineStart.x - radius, lineStart.y - radius)
        rightDown = (lineStart.x + radius, lineStart.y + radius)

        drawer.line([(lineStart.x, lineStart.y), (lineEnd.x, lineEnd.y)], fill ="red", width = 4)
        drawer.ellipse([leftUp, rightDown], fill = 'red')


def drawNode(node, image, width, height, length):

    drawer = ImageDraw.Draw(image)

    point = node.getPoint()
    padding = length * 0.05

    start = Point(point.x * length, (height - point.y - 1) * length)

    drawRect(node, start, length, drawer)

    if node.getState() == 'c' or node.getState() == 'o':
        drawCost(node, start, padding, drawer)
        drawTriangle(node, start, length, drawer)
