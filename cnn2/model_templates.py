

def template_0(input_shape):

    gradient = {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

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

    print(layers)

    return layers


def template_1(input_shape):

    gradient = {'type':'adam', 'parameter':{'lr':0.001, 'beta1':0.95, 'beta2':0.95}}

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


def createModelTemplate(input_shape, index):

    templateList = [template_0(input_shape), template_1(input_shape)]

    return templateList[index]
