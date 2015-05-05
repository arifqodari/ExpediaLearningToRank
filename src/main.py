import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from sklearn.ensemble import *


def read_data(filename,chunked):
    """
    we don't have enough memory to load a big file
    so we need to read it in chunks
    However, for some data preprocessing, no chunk will be easier.
    """

    if not chunked:
        return pd.read_csv(filename, dtype=object)
    else:
        size_in_chunk = 300000
        return pd.read_csv(filename, dtype=object, chunksize=size_in_chunk)

def read_training_data(chunked = True):
    """
    read training data
    """

    return read_data(TRAIN_DATA,chunked)

def read_test_data(chunked = True):
    """
    read test data
    """

    return read_data(TEST_DATA, chunked)

def preprocessing(df):
    """
    preprocessing data
    """

    # replace missing values with -1
    df.fillna(-1, inplace=True)

    # attribute selection
    attributes = list(df.columns)
    attributes.remove('date_time')
    attributes.remove('booking_bool')

    # create data and target matrix
    X = df[attributes].values
    y = df['booking_bool'].values

    return X, y

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


if __name__ == "__main__":
    train_reader = read_training_data()
    test_reader = read_test_data()

    df_train = train_reader.get_chunk(1000)
    df_test = test_reader.get_chunk(1000)

    X_train, y_train = preprocessing(df_train)

    clf = RandomForestClassifier(n_estimators=50,
            verbose=2,
            n_jobs=2,
            min_samples_split=10,
            random_state=1)
    #clf.fit(X_train, y_train)

