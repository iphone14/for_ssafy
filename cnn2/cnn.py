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

    data = ['train_x', 'train_y', 'test_x', 'test_y']
    shape = [train_x.shape, train_y.shape, test_x.shape, test_y.shape]
    table = {'Data':data, 'Shape':shape}
    print_table(table, True)


def print_performance(accuracy, span):

    key = ['accuracy', 'seconds']
    values = [accuracy, span.total_seconds()]
    table = {'Key':key, 'Values':values}
    print_table(table, True)


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

    epochs = 10

    train_y, test_y = oneHotEncode(train_y, test_y)

    accuracy = test(modelTemplate, epochs, train_x, train_y, test_x, test_y)

    print_performance(accuracy, (datetime.datetime.now() - start_time))


if __name__ == "__main__":
    main()
