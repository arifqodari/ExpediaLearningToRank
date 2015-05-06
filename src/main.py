import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from sklearn.ensemble import *


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

def preprocessing(df,type = 1):
    """
    preprocessing data
    Here we only drop features and transform NULL values
    After preprocessing:
        training data (type = 1): contains only training and target
        test data (type  = 0): contains only training
    No return values
    """

    # treatment for missing values
    df.orig_destination_distance.fillna(-10,inplace = True)

    # Remove srch_id and date_time
    df.drop(['srch_id','date_time'],axis = 1, inplace = True)

    # Replace NULL with -10 in place
    df.visitor_hist_starrating.fillna(-10,inplace = True)

    df.visitor_hist_adr_usd.fillna(-10,inplace = True)

    df.prop_review_score.fillna(-10, inplace = True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-350, inplace = True)

    df.prop_location_score2.fillna(0, inplace = True)

    # Replace NULL of competitiors with 0 in place
    # for i in range(1,9):
    #     rate = 'comp' + str(i) + '_rate'
    #     inv = 'comp' + str(i) + '_inv'
    #     diff = 'comp' + str(i) + '_rate_percent_diff'
    #     df[rate].fillna(0,inplace = True)
    #     df[inv].fillna(0,inplace = True)
    #     df[diff].fillna(0,inplace = True)

    # Remove all categorical attribute
    to_delete = [
        'visitor_location_country_id',
        'visitor_hist_adr_usd',
        'prop_country_id',
        'prop_id',
        'prop_brand_bool',
        'position',
        'promotion_flag',
        'srch_destination_id',
        'random_bool'
    ]
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv,diff])

    if type == 1:
        to_delete.extend(['click_bool','gross_bookings_usd'])

    df.drop(to_delete, axis = 1, inplace = True)

    # attributes = list(df.columns)
    # attributes.remove('srch_id')
    # attributes.remove('date_time')
    # attributes.remove('position')
    # attributes.remove('click_bool')
    # attributes.remove('gross_booking_usd')
    # attributes.remove('booking_bool')

    # create features matrix and target array
    # X = df[attributes].values
    # y = df['booking_bool'].values


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

