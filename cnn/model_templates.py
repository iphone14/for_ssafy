def gradient_sgd():
    return {'type':'SGD', 'parameter':{'lr':0.01}}

def gradient_adam():
    return {'type':'Adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95, 'exp':1e-7}}

def gradient_RMSprop():
    return {'type':'RMSprop', 'parameter':{'lr':0.001, 'beta':0.95, 'exp':1e-8}}



def template_complex(gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'convolution', 'parameter':{'filters':8, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'relu', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':'softmax', 'gradient':gradient}}]

    return layers


def template_light(gradient, input_shape, classes):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'relu', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':classes, 'activation':'softmax', 'gradient':gradient}}]

    return layers


def createModelTemplate(modelType, gradientType, input_shape, classes):

    modelTypeList = {'light':template_light, 'complex': template_complex}
    gradientTypeList = {'Adam':gradient_adam, 'SGD':gradient_sgd, 'RMSprop':gradient_RMSprop}

    template = modelTypeList[modelType]
    gradient = gradientTypeList[gradientType]

    return template(gradient(), input_shape, classes)
