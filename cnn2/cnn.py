from utils import *
from model import *
import datetime


def createModel1(input_shape):

    gradient = {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

    layerList = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(5, 5), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'linear', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':10, 'activation':'softmax', 'gradient':gradient}}]

    return layerList


def createModel2(input_shape):

    gradient = {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

    layerList = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'linear', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(4, 4), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':10, 'activation':'softmax', 'gradient':gradient}}]

    return layerList


def loadTrain():
    train_x, train_y = extractMNIST('./mnist/train')
    train_x = normalize(train_x)

    return train_x, train_y

def loadTest():
    test_x, test_y = extractMNIST('./mnist/test')
    test_x = normalize(test_x)

    return test_x, test_y


def main():

    train_x, train_y = loadTrain()
    test_x, test_y = loadTest()

    epochs = 100

    input_shape = train_x.shape[1:]

    start_time = datetime.datetime.now()

    model = Model(createModel2(input_shape), log='info')
    model.build()
    model.train(train_x, train_y, epochs)
    prediction = model.predict(test_x)

    print('---------------------Predict---------------------')

    count = len(prediction)
    correct = 0

    for i in range(count):
        pred = np.argmax(prediction[i])
        if pred == test_y[i]:
            correct += 1
            print('predict={0:12} correct={1}'.format((str(test_y[i]) +'/' + str(pred)), 'O'))
        else:
            print('predict={0:12} correct={1}'.format((str(test_y[i]) +'/' + str(pred)), 'X'))

    end_time = datetime.datetime.now()
    accuracy = float(correct / count) * 100
    print('---------------------Summary---------------------')
    print('accuracy={0}%'.format(accuracy))
    print('epochs={0}'.format(epochs))
    print('data_count={0}'.format(len(train_x)))
    print('seconds={0}'.format((end_time - start_time).total_seconds()))



if __name__ == "__main__":
    main()
