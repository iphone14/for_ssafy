from utils import *
from model import *
from model_templates import *
import datetime as dt
import argparse


def makeOneHotMap(train_y, test_y):

    labels = np.hstack((train_y, test_y))

    unique = np.unique(labels, return_counts=False)

    return {key : index for index, key in enumerate(unique)}


def oneHotEncode(oneHotMap, train_y, test_y):

    labels = np.hstack((train_y, test_y))

    classes = len(oneHotMap)

    labels = [np.eye(classes)[oneHotMap[y]].reshape(classes, 1) for y in labels]
    labels = np.array(labels)

    train = labels[0:len(train_y)]
    test = labels[-len(test_y):]

    return train, test


def loadTrain(datasetType):
    train_x, train_y = extractMNIST('./mnist/' + datasetType + '/train')
    train_x = normalize(train_x)

    return train_x, train_y


def loadTest(datasetType):
    test_x, test_y = extractMNIST('./mnist/' + datasetType + '/test')
    test_x = normalize(test_x)

    return test_x, test_y


def print_oneHotMap(oneHotMap):

    oneHotList = []
    labelList = []

    classes = len(oneHotMap)

    for mapKey in oneHotMap:
        map = np.eye(classes)[oneHotMap[mapKey]].reshape(classes, 1)
        oneHotList.append(map.reshape(-1))
        labelList.append(mapKey)

    print_table({'Label':labelList, 'OneHot':oneHotList}, True)


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


def print_config(model, gradient, epochs, dataset):

    sizeFullText = {'sm': 'small', 'md':'medium', 'lg':'large'}

    Config = ['model', 'gradient', 'epochs', 'dataset']
    values = [sizeFullText[model], gradient, epochs, sizeFullText[dataset]]
    table = {'Config':Config, 'Values':values}
    print_table(table, True)


def test(modelTemplate, epochs, train_x, train_y, test_x, test_y):

    model = Model(modelTemplate, log='info')
    model.build()
    model.train(train_x, train_y, epochs)
    accuracy = model.test(test_x, test_y)

    return accuracy


def main(modelType, gradientType, epochs, datasetType):

    start_time = dt.datetime.now()

    train_x, train_y = loadTrain(datasetType)
    test_x, test_y = loadTest(datasetType)

    print_shapes(train_x, train_y, test_x, test_y)

    modelTemplate = createModelTemplate(modelType, gradientType, train_x.shape[1:])

    oneHotMap = makeOneHotMap(train_y, test_y)

    print_oneHotMap(oneHotMap)

    train_y, test_y = oneHotEncode(oneHotMap, train_y, test_y)

    accuracy = test(modelTemplate, epochs, train_x, train_y, test_x, test_y)

    print_performance(accuracy, (dt.datetime.now() - start_time))


def parse_arg():
    parser = argparse.ArgumentParser(prog='CNN')
    parser.add_argument('-m', dest='modelType', type=str, default='sm', choices=['sm', 'md', 'lg'], help='sample model type (default:lg)')
    parser.add_argument('-g', dest='gradientType', type=str, default='adam', choices=['adam', 'sgd', 'RMSprop'], help='sample gradient type (default: adam)')
    parser.add_argument('-e', dest='epochs', type=int, default=50, help='epochs (default: 50)')
    parser.add_argument('-d', dest='datasetType', type=str, default='sm', choices=['sm', 'md', 'lg'], help='train set size (default: sm)')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arg()
    print_config(args.modelType, args.gradientType, args.epochs, args.datasetType)
    main(args.modelType, args.gradientType, args.epochs, args.datasetType)
