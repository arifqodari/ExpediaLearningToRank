import numpy as np
import pandas as pd
import cPickle as pickle
import os.path

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


def save_model(clf, model_filename):
    """
    save classifier to file
    """

    output = open(folder+model_filename+'.pickle', 'w')
    pickle.dump(clf, output)


def load_model(model_filename):
    """
    load classifier from file
    """

    inp = open(folder+model_filename+'.pickle', 'r')
    return pickle.load(inp)


def save_var(var, var_name):
    """
    save variable
    """

    output = open(folder+var_name+'.pickle', 'w')
    pickle.dump(var, output)


def load_var(var_name):
    """
    load variable
    """

    inp = open(folder+var_name+'.pickle', 'r')
    return pickle.load(inp)

def is_file_exist(var_name):
    return os.path.isfile(folder+var_name+'.pickle')
