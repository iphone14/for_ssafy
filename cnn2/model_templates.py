
def gradient_adam():
    return {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

def gradient_sgd():
    return {'type':'sgd', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

def gradient_RMSprop():
    return {'type':'RMSprop', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}


def template_lg(gradient, input_shape):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(5, 5), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'linear', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':5, 'activation':'linear', 'gradient':gradient}}]

    return layers


def template_md(gradient, input_shape):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(5, 5), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'linear', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(2, 2), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':5, 'activation':'linear', 'gradient':gradient}}]

    return layers


def template_sm(gradient, input_shape):

    layers = [
        {'type':'input', 'parameter':{'input_shape':input_shape}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'linear', 'gradient':gradient}},
        {'type':'convolution', 'parameter':{'filters':3, 'kernel_size':(3, 3), 'strides':(1, 1), 'padding':True, 'activation':'relu', 'gradient':gradient}},
        {'type':'maxPooling', 'parameter':{'pool_size':(4, 4), 'strides':None}},
        {'type':'flatten', 'parameter':{}},
        {'type':'dense', 'parameter':{'units':128, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':64, 'activation':'linear', 'gradient':gradient}},
        {'type':'dense', 'parameter':{'units':5, 'activation':'linear', 'gradient':gradient}}]

    return layers


def createModelTemplate(modelType, gradientType, input_shape):

    modelTypeList = {'sm': template_sm, 'md': template_md, 'lg':template_lg}
    gradientTypeList = {'adam':gradient_adam, 'sgd':gradient_sgd, 'RMSProp':gradient_RMSprop}

    template = modelTypeList[modelType]
    gradient = gradientTypeList[gradientType]

    return template(gradient(), input_shape)
