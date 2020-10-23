from forward import *
from backward import *
from utils import *

import numpy as np
import pickle
from tqdm import tqdm

import os

from array import *
from random import shuffle


def conv(image, label, params):

    conv_stride = 1
    pool_stride = 2

    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    ################################################
    ############## Forward Operation ###############
    ################################################

    conv1 = convolution(image, f1, b1, conv_stride) # convolution operation
    conv1[conv1<=0] = 0 # pass through ReLU non-linearity


    conv2 = convolution(conv1, f2, b2, conv_stride) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_stride) # maxpooling operation

    #print(pooled.shape)


    (nf2, dim2, _) = pooled.shape
    #print(dim2)
    #print(nf2 * dim2 * dim2)
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer


    #print("fc : ", fc.shape)
    #print('w3 : ', w3.shape)
    #print('b3 : ', b3.shape)


    z = w3.dot(fc) + b3 # first dense layer

    out = w4.dot(z) + b4 # second dense layer

    probs = softmax(out) # predict class probabilities with the softmax activation function

    ################################################
    #################### Loss ######################
    ################################################

    loss = categoricalCrossEntropy(probs, label) # categorical cross-entropy loss

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output

    #print('ege', label)
    dw4 = dout.dot(z.T) # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis = 1).reshape(b4.shape) # loss gradient of final dense layer biases

    dz = w4.T.dot(dout) # loss gradient of first dense layer outputs
    dz[z<=0] = 0 # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis = 1).reshape(b3.shape)

    dfc = w3.T.dot(dz) # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer

    dconv2 = maxpoolBackward(dpool, conv2, pool_stride) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2[conv2<=0] = 0 # backpropagate through ReLU

    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_stride) # backpropagate previous gradient through second convolutional layer.
    dconv1[conv1<=0] = 0 # backpropagate through ReLU

    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_stride) # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss

#####################################################
################### Optimization ####################
#####################################################

def adamGD(X, Y, num_classes, dim, params, cost, paramsAdam):

    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    [v1, v2, v3, v4, bv1, bv2, bv3, bv4] = paramsAdam

    #print(X.shape)
    size = len(X)
    #X = X.reshape(size, n_c, dim, dim)

    cost_ = 0

    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)


    #size is image count


    for i in range(size):

        x = X[i]

        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot

        # Collect Gradients for training example
        grads, loss = conv(x, y, params)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_

        cost_+= loss

    # Parameter Update



    #f1 -= (lr * (df1 / size))
    #b1 -= (lr * (db1 / size))

    #f2 -= (lr * (df2 / size))
    #b2 -= (lr * (db2 / size))

    #w3 -= (lr * (dw3 / size))
    #b3 -= (lr * (db3 / size))

    #w4 -= (lr * (dw4 / size))
    #b4 -= (lr * (db4 / size))

    lr = 0.001
    beta1 = 0.95


    v1 = beta1 * v1 + (1 - beta1) * (df1 / size)**2
    f1 -= lr * (df1 / size)/(np.sqrt(v1) + 1e-7)

    bv1 = beta1 * bv1 + (1 - beta1) * (db1 / size)**2
    b1 -= lr * (db1 / size)/(np.sqrt(bv1) + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * (df2 / size)**2
    f2 -= lr * (df2 / size)/(np.sqrt(v2) + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * (db2 / size)**2
    b2 -= lr * (db2 / size)/(np.sqrt(bv2) + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * (dw3 / size)**2
    w3 -= lr * (dw3 / size)/(np.sqrt(v3) + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * (db3 / size)**2
    b3 -= lr * (db3 / size)/(np.sqrt(bv3) + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * (dw4 / size)**2
    w4 -= lr * (dw4 / size)/(np.sqrt(v4) + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * (db4 / size)**2
    b4 -= lr * (db4 / size) / (np.sqrt(bv4) + 1e-7)

    cost_ = cost_/size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    paramsAdam = [v1, v2, v3, v4, bv1, bv2, bv3, bv4]

    return params, cost, paramsAdam

#####################################################
##################### Training ######################
#####################################################
def train(num_classes = 10, img_dim = 28, f = 5, num_filt1 = 3, num_filt2 = 3):

    X, Y = extractMNIST('./mnist/training')

    X-= int(np.mean(X))
    X/= int(np.std(X))

    #train_data = np.hstack((X, y_dash))



    ## Initializing all the parameters
    f1, f2, w3, w4 = (num_filt1 ,X.shape[1], f, f), (num_filt2 ,num_filt1, f, f), (128, 300), (10, 128)

    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    paramsAdam = [v1, v2, v3, v4, bv1, bv2, bv3, bv4]

    cost = []

    epochs = 50

    for epoch in range(epochs):
        params, cost, paramsAdam = adamGD(X, Y, num_classes, img_dim, params, cost, paramsAdam)
        print(cost[-1])
        #t.set_description("Cost: %.2f" % (cost[-1]))
        #a = a + 1
        #print(a)

    return params, cost
