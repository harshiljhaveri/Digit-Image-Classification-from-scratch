# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from collections import defaultdict

# Importing training set and testing set from cmd
training_set = sys.argv[1]  # 'train_image.csv'
training_labels = sys.argv[2]  # 'train_label.csv'
testing_set = sys.argv[3]  # 'test_image.csv'

# read csv file and convert to numpy
training_set = pd.read_csv(training_set, sep=',', header=None).values
training_labels = pd.read_csv(training_labels, sep=',', header=None).values
testing_set = pd.read_csv(testing_set, sep=',', header=None).values
# constructing images requires reshaping to 2d aarray
training_set_2d = np.reshape(training_set, (np.shape(training_set)[0], 28, 28))
testing_set_2d = np.reshape(testing_set, (np.shape(testing_set)[0], 28, 28))

# until the 10000th index all images are sliced
training_set = training_set[:10000]
training_set_len = len(training_set)
training_labels = (training_labels[:10000])
# testing_set = testing_set.T

# Augmenting the datast for better accuracy
training_set = np.concatenate((training_set, training_set), axis=0)
training_labels = np.concatenate((training_labels, training_labels), axis=0)
training_set = np.concatenate((training_set, training_set[:10000]), axis=0)
training_labels = np.concatenate((training_labels, training_labels[:10000]), axis=0)

# Using the shift method on all images
for i in range(training_set_len, 2 * training_set_len):
    img = training_set[i]
    label = training_labels[i]
    image = img.reshape((28, 28))
    shifted_image = shift(image, [2, 2], cval=0, mode="constant")
    training_set[i] = shifted_image.flatten()

# Using the rotate method on all numbers upto a minor degree of 10
for i in range(2 * training_set_len, 3 * training_set_len):
    img = training_set[i]
    label = training_labels[i]
    ro = rotate(img.reshape(28, 28).astype(np.uint8), 10, reshape=False)
    training_set[i] = ro.flatten()

examples = len(training_labels)


# numpy version of to_categorical method of keras for one hot encoding of labels
def categorical(train_labels):
    labs = np.zeros((train_labels.shape[0], 10))
    for i in range(examples):
        labs[i][train_labels[i]] = 1
    return labs


training_labels = categorical(training_labels)

layer = defaultdict()
# defining the dimensions of the three layers
# first layer has 512 nodes and size of each equal to the image dimension
layer[1] = (512, training_set.shape[1])
# hidden layer contains 256 nodes downsampling and size equal to no. of nodes in l1
layer[2] = (256, layer[1][0])
# output layer contains nodes equal to number of classes
layer[3] = (10, layer[2][0])
actv= defaultdict()
output = defaultdict()


# print(training_set.shape)
# print(training_labels.shape)
# print(testing_set.shape)

# avoiding the classic problem of overflow in sigmoid function
def prevent_overflow(x):
    return np.clip(x, -layer[1][0], layer[1][0])


# defining sigmoid activation function with prevention of overflow
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-prevent_overflow(x)))


# defning the relu activation function
def relu(x):
    return np.max(0, x)


# computing the cost after each forward propagation
def categorical_cross_entropy(A, Y):
    m = Y.shape[1]
    cross_entropy = np.dot((-1.0 / m), sum((Y* np.log10(A)) +((1 - Y) * np.log10(1 - A))))
    cost = 2 * (A - Y)
    return cost


# computing forward propagation for computing output values out of the given weights and biases
def forward_propagation(inp, w, b, Y):
    # input examples will be initial imported array
    actv[0] = inp
    # update all layers using sigmoid act function
    for i in range(1, 3):
        output[i] = np.add(np.matmul(w[i], actv[i - 1]), b[i])
        actv[i] = sigmoid(output[i])
    # updating the final layer with softmax instead of sigmoid
    output[3] = np.add(np.matmul(w[3], actv[2]), b[3])
    actv[3] = np.exp(output[3] - output[3].max()) / np.sum(np.exp(output[3] - output[3].max()), axis=0)
    
    # in case the for prop is being carried out for test set, then error can not be computed since we dont have the labels
    if Y == 'test':
        Y = np.zeros(actv[3].shape)
    # compute error for calculated predictions
    error = categorical_cross_entropy(actv[3], Y)
    return output, actv, error


def gradient_descent(x, grad_x, lr):
    # update weights and biases according to value of differential gradient
    x -= np.multiply(lr, grad_x)
    return x


def back_propagation(w, b, inp, actv, error, lr, examples_here):
    # calculating gradient weight and bias for third layer
    grad_w3 = np.multiply((1.0 / examples_here), np.matmul(error, np.transpose(actv[2])))
    grad_b3 = np.multiply((1.0 / examples_here), np.sum(error, axis=1, keepdims=True))
    # send third layer weights and bias for updating
    w[3] = gradient_descent(w[3], grad_w3, lr)
    b[3] = gradient_descent(b[3], grad_b3, lr)

    # calculating gradient activation and output for second layer
    grad_a2 = np.matmul(np.transpose(w[3]), error)
    grad_op2 = grad_a2 * actv[2] * (1 - actv[2])
    # calculating gradient weight and bias for second layer
    grad_w2 = np.multiply((1.0 / examples_here), np.matmul(grad_op2, np.transpose(actv[1])))
    grad_b2 = np.multiply((1.0 / examples_here), np.sum(grad_op2, axis=1, keepdims=True))
    # send second layer weights and bias for updating
    w[2] = gradient_descent(w[2], grad_w2, lr)
    b[2] = gradient_descent(b[2], grad_b2, lr)

    # calculating gradient activation and output for first layer
    grad_a1 = np.matmul(np.transpose(w[2]), grad_op2)
    grad_op1 = grad_a1 * actv[1] * (1 - actv[1])
    # calculating gradient weight and bias for first layer
    grad_w1 = np.multiply((1.0 / examples_here), np.matmul(grad_op1, np.transpose(inp)))
    grad_b1 = np.multiply((1.0 / examples_here), np.sum(grad_op1, axis=1, keepdims=True))
    # send first layer weights and bias for updating
    w[1] = gradient_descent(w[1], grad_w1, lr)
    b[1] = gradient_descent(b[1], grad_b1, lr)
    # return updated weights and biases
    return w, b


def train(training_set, training_labels, testing_set, examples, layer, batch_size=32,
          lr=0.01, epochs=50):
    w = defaultdict()
    b = defaultdict()
    # initialises the primary parameters
    for i in range(1, 4):
        # weights are initialised randomly
        w[i] = np.multiply(np.sqrt(1.0 / layer[i][1]), np.random.randn(layer[i][0], layer[i][1]))
        # bias is innitialised to 0
        b[i] = np.zeros((layer[i][0], 1))

    for k in range(epochs):
        # print('epoch:', k)
        # in order to avoid overfitting, input is shuffled at each epoch so that weights dont get biased
        shuf = np.array([j for j in range(examples)])
        np.random.shuffle(shuf)
        # same order needs to be maintained for images and labels
        training_set = training_set[:, shuf]
        training_labels = training_labels[:, shuf]
        img_batches = {}
        label_batches = {}
        # Stochastic Gradient Descent
        mod = 0 if (examples % batch_size == 0) else 1
        batch_iter = (examples // batch_size) + mod
        # stochastic implies dividing input into batches of smaller size
        # print('bi:',(batch_iter))
        for i in range(batch_iter):
            # print('batchno:', i)
            # now for each batch get the images and labels
            img_batches[i] = training_set[:, i * batch_size: (i + 1) * batch_size]
            label_batches[i] = training_labels[:, i * batch_size: (i + 1) * batch_size]
            # get input length
            examples_here = np.shape(img_batches[i])[1]
            # feed the given input to feedforward network
            output, actv, error = forward_propagation(img_batches[i], w, b, label_batches[i])
            # update weights and biases using backprop
            w, b = back_propagation(w, b, img_batches[i], actv, error,
                                    lr, examples_here)
    # once the model has been trained perfectly, now pass the test set thru feedforward network
    output, actv, error = forward_propagation(testing_set, w, b, 'test')
    # save output predicted to output file
    pd.DataFrame(np.argmax(actv[3], axis=0)).to_csv('test_predictions.csv', header=False, index=False)

# run the given network now, hyperparameters are set as batch size 32, learning rate 10^-2 and 50 epochs
train(np.transpose(training_set), np.transpose(training_labels), np.transpose(testing_set), examples, layer, 32, 0.015,
      50)

# to make sure the labels and image are perfectly synced during randomisation, cross check using display image function
def display_img(arr):
    plt.imshow(arr, cmap='gray')
    plt.show()

# display_img(np.reshape(training_set[:, 20000], (28, 28)))
# print(np.argmax(training_labels[:,20000]))
