import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from data_reader import *

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
    print "The preprocessing task is done."
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


def simple_preprocessing(df):
    """
    simple pre-processing
    """

    # treatment for missing values
    df.orig_destination_distance.fillna(-10,inplace = True)

    df.visitor_hist_starrating.fillna(-10,inplace = True)

    df.visitor_hist_adr_usd.fillna(-10,inplace = True)

    df.prop_review_score.fillna(-10, inplace = True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-350, inplace = True)

    df.prop_location_score2.fillna(0, inplace = True)

    # Replace NULL of competitiors with 0 in place
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        df[rate].fillna(0,inplace = True)
        df[inv].fillna(0,inplace = True)
        df[diff].fillna(0,inplace = True)

    attributes = list(df.columns)
    attributes.remove('srch_id')
    attributes.remove('date_time')
    attributes.remove('position')
    attributes.remove('click_bool')
    attributes.remove('gross_bookings_usd')
    attributes.remove('booking_bool')

    # create features matrix and target array
    X = df[attributes].astype(float).values
    y = df['booking_bool'].astype(int).values

    return X, y
