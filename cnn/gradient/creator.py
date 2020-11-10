from gradient.adam import *
from gradient.rms_prop import *
from gradient.sgd import *

def createGradient(gradient):

    type = gradient['type']
    parameter = gradient['parameter']
    typeClass = {'RMSprop':RMSprop, 'Adam':Adam, 'SGD':SGD}

    return typeClass[type](**parameter)
