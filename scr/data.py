# scr/data.py
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt


def load_and_preprocess():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # standardize (z-score)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)

    # reshape
    X_train = X_train.reshape((-1,28,28,1))
    X_test = X_test.reshape((-1,28,28,1))

    #labels -> one-hot encoding
    num_classes = 10
    y_train = to_categorical(y_train,num_classes)
    y_test = to_categorical(y_test,num_classes)

    return X_train,y_train,X_test,y_test



