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
    DTYPE_DICT = {
        'comp1_inv': 'float16',
        'comp1_rate': 'float16',
        'comp1_rate_percent_diff': 'float16',
        'comp2_inv': 'float16',
        'comp2_rate': 'float16',
        'comp2_rate_percent_diff': 'float16',
        'comp3_inv': 'float16',
        'comp3_rate': 'float16',
        'comp3_rate_percent_diff': 'float16',
        'comp4_inv': 'float16',
        'comp4_rate': 'float16',
        'comp4_rate_percent_diff': 'float16',
        'comp5_inv': 'float16',
        'comp5_rate': 'float16',
        'comp5_rate_percent_diff': 'float16',
        'comp6_inv': 'float16',
        'comp6_rate': 'float16',
        'comp6_rate_percent_diff': 'float16',
        'comp7_inv': 'float16',
        'comp7_rate': 'float16',
        'comp7_rate_percent_diff': 'float16',
        'comp8_inv': 'float16',
        'comp8_rate': 'float16',
        'comp8_rate_percent_diff': 'float16',
        'date_time': 'str',
        'orig_destination_distance': 'float16',
        'price_usd': 'float16',
        'promotion_flag': 'int8',
        'prop_brand_bool': 'int8',
        'prop_country_id': 'int8',
        'prop_id': 'int64',
        'prop_location_score1': 'float16',
        'prop_location_score2': 'float16',
        'prop_log_historical_price': 'float16',
        'prop_review_score': 'float16',
        'prop_starrating': 'float16',
        'random_bool': 'int8',
        'site_id': 'int8',
        'srch_adults_count': 'int16',
        'srch_booking_window': 'int16',
        'srch_children_count': 'int16',
        'srch_destination_id': 'int8',
        'srch_id': 'int',
        'srch_length_of_stay': 'int16',
        'srch_query_affinity_score': 'float16',
        'srch_room_count': 'int16',
        'srch_saturday_night_bool': 'int8',
        'visitor_hist_adr_usd': 'float16',
        'visitor_hist_starrating': 'float16',
        'visitor_location_country_id': 'int8'
        }
    if chunked is None:
        chunked = 300000

    if chunked:
        return pd.read_csv(filename, dtype=DTYPE_DICT, chunksize=chunked,sep = ',')
    else:
        return pd.read_csv(filename, dtype=DTYPE_DICT,sep = ',')


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


def save_data(df, type = 1,chunked = True):
    """
    save data into csv file
    type = 1: training data 
    type = 0: test data
    If for the chunked data, header will not be written
    """

    df.to_csv(PROCESSED_TRAIN if type == 1 else PROCESSED_TEST,sep=',',mode = 'a+' if chunked else 'w',index = False,header = False if chunked else True)


def save_sampled_data(object):
    """
    Save Sampled Training Data into file.
    """

    output = open(SAMPLED_TRAIN,'w')
    pickle.dump(object,output)
    output.close()


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
