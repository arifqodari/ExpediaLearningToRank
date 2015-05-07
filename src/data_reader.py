import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *


def read_data(filename, chunked):
    """
    we don't have enough memory to load a big file
    so we need to read it in chunks
    However, for some data preprocessing, no chunk will be easier.
    chunked = False, to disable chunking data
    """

    if chunked is None:
        chunked = 300000

    if chunked:
        return pd.read_csv(filename, dtype=object, chunksize=chunked)
    else:
        return pd.read_csv(filename, dtype=object)


def read_training_data(chunked=None):
    """
    read training data
    """

    return read_data(TRAIN_DATA, chunked)


def read_kaggle_training_data(chunked = None):
    """
    read kaggle training data
    """

    return read_data(KAGGLE_TRAIN_DATA,chunked)


def read_test_data(chunked = None):
    """
    read test data
    """

    return read_data(TEST_DATA, chunked)


def save_data(tfr, type = 1):
    """
    save data into file
    type = 1: training data 
    type = 0: test data
    tfr is short for text file reader
    """
    if type == 1:
        output = open(PROCESSED_TRAIN,'w')
    else:
        output = open(PROCESSED_TEST,'w')

    pickle.dump(tfr,output)


def save_model(clf):
    """
    save classifier to file
    """

    output = open(MODEL_FILENAME, 'w')
    pickle.dump(clf, output)


def load_model():
    """
    load classifier from file
    """

    inp = open(MODEL_FILENAME, 'r')
    return pickle.load(inp)
