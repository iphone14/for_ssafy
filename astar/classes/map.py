from classes.node import Node
from classes.point import Point
import classes.drawer as drawer


class Map:
    def __init__(self, data, width, height, stop):

        self.map = []
        self.width = width
        self.height = height

        index = 0

        for d in data:
            x = index % self.width
            y = height - 1 - (index // self.width)
            point = Point(x, y)
            h = (abs(stop.x - point.x) + abs(stop.y - point.y)) * 10

            node = Node(point, h)

            if d == -1:
                node.setBlock()

            self.map.append(node)

            index += 1

    def getNode(self, point):
        if point.x < 0 or point.y < 0:
            print('over', point.y, ', ', point.x)
            return None

        if point.x < self.width and point.y < self.height:
            index = point.x + ((self.width * self.height) - ((point.y + 1) * self.width))
            return self.map[index]
        else:
            print('over', point.y, ', ', point.x)
            return None

    def show(self):

        drawer.draw(self)
        text = []
        text2 = []
        text3 = []

        for y in range(self.height):
            line = []
            line2 = []
            line3 = []
            for x in range(self.width):
                node = self.getNode(Point(x, self.height - y - 1))
                #line.append(repr(node.getPoint()))
                #line.append(str(node.costG())+ ",  ")

                line.append(str(node.getState()) + ",  ")

                if node.getState() == 'o':
                    line2.append(str(node.costG()) + ",  ")
                else:
                    line2.append(str(node.getState()) + ",  ")

                line3.append(str(node.costF()) + ",  ")

            line.append('\n')
            line2.append('\n')
            line3.append('\n')

            text += line
            text2 += line2

        """print("".join(text))"""
        print("".join(text2))
