from utils import *
from model import *
from model_templates import *
import datetime


def oneHotEncode(train_y, test_y):

    labels = np.hstack((train_y, test_y))

    unique = np.unique(labels, return_counts=False)

    labelIndexs = {key : index for index, key in enumerate(unique)}
    classes = len(labelIndexs)

    labels = [np.eye(classes)[labelIndexs[y]].reshape(classes, 1) for y in labels]
    labels = np.array(labels)

    train = labels[0:len(train_y)]
    test = labels[-len(test_y):]

    return train, test


def loadTrain():
    train_x, train_y = extractMNIST('./mnist/train')
    train_x = normalize(train_x)

    return train_x, train_y


def loadTest():
    test_x, test_y = extractMNIST('./mnist/test')
    test_x = normalize(test_x)

    return test_x, test_y


def print_shapes(train_x, train_y, test_x, test_y):

    print('---------------------Shape---------------------')
    print('train_x{0:14}shape={1}'.format('', train_x.shape))
    print('train_y{0:14}shape={1}'.format('', train_y.shape))
    print('test_x{0:15}shape={1}'.format('', test_x.shape))
    print('test_y{0:15}shape={1}'.format('', test_y.shape))


def print_performance(accuracy, span):

    print('---------------------Performance---------------------')
    print('accuracy{0:13}{1}%'.format('', accuracy))
    print('seconds{0:14}{1}'.format('', span.total_seconds()))


def test(modelTemplate, epochs, train_x, train_y, test_x, test_y):

    model = Model(modelTemplate, log='info')
    model.build()
    model.train(train_x, train_y, epochs)
    accuracy = model.test(test_x, test_y)

    return accuracy


def main():

    start_time = datetime.datetime.now()

    train_x, train_y = loadTrain()
    test_x, test_y = loadTest()

    print_shapes(train_x, train_y, test_x, test_y)

    modelTemplate = createModelTemplate(train_x.shape[1:], 0)

    epochs = 1

    train_y, test_y = oneHotEncode(train_y, test_y)

    accuracy = test(modelTemplate, epochs, train_x, train_y, test_x, test_y)

    print_performance(accuracy, (datetime.datetime.now() - start_time))


if __name__ == "__main__":
    main()
