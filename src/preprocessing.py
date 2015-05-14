import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from data_reader import *

# Scheme 1: Remove all categorical attributes
def preprocessing_1(df,type = 1):
    """
    Preprocessing data
    Here we only drop features and transform NULL values
    After preprocessing:
        training data (type = 1): contains only training and target
        test data (type  = 0): contains only training
    No return values since all operation is done in place
    """

    # treatment for missing values
    df.orig_destination_distance.fillna(-10,inplace = True)

    # Replace NULL with -10 in place
    df.visitor_hist_starrating.fillna(-10,inplace = True)

    #df.visitor_hist_adr_usd.fillna(-10,inplace = True)

    df.prop_review_score.fillna(-10, inplace = True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-10, inplace = True)

    df.prop_location_score2.fillna(-10, inplace = True)

    # Remove all categorical attribute
    to_delete = [
        'date_time',
        'site_id',
        'visitor_location_country_id',
        'visitor_hist_adr_usd',
        'prop_country_id',
        #'prop_id',
        'prop_brand_bool',
        'promotion_flag',
        'srch_destination_id',
        'random_bool'
    ]
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv,diff])

    #This goes for the training data
    if type == 1:
        to_delete.extend(['position','gross_bookings_usd'])

    df.drop(to_delete, axis = 1, inplace = True)

    """
    Average of numeric attributes per prop_id
    Generate the average data here.
    avg_propId_train and avg_propId_test
    """



    print "The preprocessing task is done."




# Scheme 2: Keep all categorical attributes 
# (except srch_id and data_time)
def preprocessing_2(df,type = 1):
    """
    preprocessing data
    Here we only drop features and transform NULL values
    After preprocessing:
        training data (type = 1): contains only training and target
        test data (type  = 0): contains only training
    No return values since all operation is done in place
    """

    # Remove date_time
    df.drop('date_time',axis = 1, inplace = True)


    # treatment for missing values
    df.orig_destination_distance.fillna(-10,inplace = True)

    # Replace NULL with -10 in place
    df.visitor_hist_starrating.fillna(-10,inplace = True)

    df.visitor_hist_adr_usd.fillna(-10,inplace = True)

    df.prop_review_score.fillna(-10, inplace = True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-350, inplace = True)

    df.prop_location_score2.fillna(0, inplace = True)

    #Replace NULL of competitiors with 0 in place
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        df[rate].fillna(0,inplace = True)
        df[inv].fillna(0,inplace = True)
        df[diff].fillna(0,inplace = True)

    #This only goes for training data
    if type == 1:
        df.drop(['srch_id','position','gross_bookings_usd'], axis = 1, inplace = True)

    print "The preprocessing task is done."

# Add average numeric attributes to the training and test data
# Average is based on training and test data at the same time
# def add_numeric_per_propid(df,avg_propId_train,avg_propId_test){
#     trainingData = read_
# }


def preprocessing_3(df, type = 1):
    """
    Preprocessing data
    Here we only drop features and transform NULL values
    After preprocessing:
        training data (type = 1): contains only training and target
        test data (type  = 0): contains only training
    No return values since all operation is done in place
    """
    to_delete = [
        'date_time',
        'site_id',
        'visitor_location_country_id',
        #'visitor_hist_adr_usd',
        'prop_country_id',
        #'prop_id',
        #'prop_brand_bool',
        #'promotion_flag',
        'srch_destination_id',
        #'random_bool',
    ]

    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        #diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv])

    #This goes for the training data
    if type == 1:
        to_delete.extend(['position','gross_bookings_usd'])

    df.drop(to_delete, axis = 1, inplace = True)
    df.fillna(-10, inplace = True)

    print "The preprocessing task is done."








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
