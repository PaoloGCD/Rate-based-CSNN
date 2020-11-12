'''
Data preprocessing
Extended from Peter Diehl's implementation https://github.com/peter-u-diehl/stdp-mnist.git

@author: Paolo G. Cachi
'''


import os
import time
import numpy as np

import cPickle as pickle

from struct import unpack


MNIST_data_path = '../../dataset/'


def process_dataset(dataset_name, bTrain=True):

    # Open the images with gzip in read binary mode
    if bTrain:
        images = open(MNIST_data_path + 'train-images-idx3-ubyte', 'rb')
        labels = open(MNIST_data_path + 'train-labels-idx1-ubyte', 'rb')
    else:
        images = open(MNIST_data_path + 't10k-images-idx3-ubyte', 'rb')
        labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte', 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = unpack('>I', images.read(4))[0]
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = unpack('>I', labels.read(4))[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
    for i in xrange(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)] for unused_row in xrange(rows)]
        y[i] = unpack('>B', labels.read(1))[0]

    np.save(dataset_name + '-samples', x)
    np.save(dataset_name + '-labels', y)


# PROCESS DATA SET

# training data set
start = time.time()
training = process_dataset(MNIST_data_path + 'fashionMNIST-training')
end = time.time()
print 'time needed to load training set:', end - start

# test data set
start = time.time()
testing = process_dataset(MNIST_data_path + 'fashionMNIST-testing', bTrain=False)
end = time.time()
print 'time needed to load test set:', end - start