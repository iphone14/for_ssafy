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


def loadDataSet(classes):
    train_x, train_y, test_x, test_y = extractMNIST(classes, './mnist/train', './mnist/test')

    train_x = normalize(train_x)
    test_x = normalize(test_x)

    return train_x, train_y, test_x, test_y

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

    performance = ['accuracy', 'minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [str(accuracy) + ' %', min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table, True)


def print_arg(model, gradient, classes, epochs, batches, train_dataset_len):

    reduced = batches > train_dataset_len

    batches = train_dataset_len if reduced else batches

    batch_str = str(batches) + (' (reduced)' if reduced else '')

    arg = ['classes', 'model', 'gradient', 'epochs', 'train dataset length', 'batches']
    values = [classes, model, gradient, epochs, train_dataset_len, batch_str]
    table = {'Argument':arg, 'Values':values}
    print_table(table, True)


def test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches):

    model = Model(modelTemplate, log='info')
    model.build()
    model.train(train_x, train_y, epochs, batches)
    accuracy = model.test(test_x, test_y)

    return accuracy


def adjust_batches(batches, train_dataset_len):
    return batches if train_dataset_len > batches else train_dataset_len


def main(modelType, gradientType, classes, epochs, batches):

    start_time = dt.datetime.now()

    train_x, train_y, test_x, test_y = loadDataSet(classes)

    print_arg(modelType, gradientType, classes, epochs, batches, len(train_x))

    batches = adjust_batches(batches, len(train_x))

    print_shapes(train_x, train_y, test_x, test_y)

    oneHotMap = makeOneHotMap(train_y, test_y)

    print_oneHotMap(oneHotMap)

    train_y, test_y = oneHotEncode(oneHotMap, train_y, test_y)

    modelTemplate = createModelTemplate(modelType, gradientType, train_x.shape[1:], len(oneHotMap))

    accuracy = test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches)

    print_performance(accuracy, (dt.datetime.now() - start_time))


def parse_arg():

    parser = argparse.ArgumentParser(prog='CNN')
    parser.add_argument('-c', dest='classes', type=int, default='3', metavar="[1-10]", help='classes (default: 3)')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'], help='sample gradient type (default: RMSprop)')
    parser.add_argument('-e', dest='epochs', type=int, default=50, help='epochs (default: 50)')
    parser.add_argument('-b', dest='batches', type=int, help='batches (default: classes x 4)')

    args = parser.parse_args()

    if args.classes < 1 or args.classes > 10:
        print('CNN: error: argument -c: invalid value: ', str(args.classes), ' (value must be 1 from 10')
        return None

    if args.batches == None:
        args.batches = args.classes * 4


    print(args.batches)

    if args.batches < 1:
        print('CNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0')
        return None





    return args

if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.gradientType, args.classes, args.epochs, args.batches)
