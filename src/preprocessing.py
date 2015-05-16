import numpy as np
import pandas as pd
import cPickle as pickle
import csv

from itertools import groupby
from sets import Set
from setting import *
from data_reader import *
from sklearn import preprocessing

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








def pointwise_label(table):
    """
    generate label for 2 classifiers
    based on click_bool and book_bool
    """

    y = np.zeros(table.shape, dtype=int)

    for idx, row in enumerate(table):

        # rel 0
        if row[0] == 0 and row[1] == 0:
            y[idx,0] = 1
            y[idx,1] = 1

        # rel 1
        elif row[1] == 1 and row[1] == 0:
            y[idx,0] = 0
            y[idx,1] = 1

        # rel 5
        # else:
        #     y[idx,0] = 0
        #     y[idx,1] = 0

    return y

def pointwise_relevance(table):
    """
    generate label for regressor
    based on click_bool and book_bool
    """

    y = np.zeros(table.shape[0], dtype=int)

    for idx, row in enumerate(table):

        # rel 0
        if row[0] == 0 and row[1] == 0:
	    y[idx] = 0

        # rel 1
        elif row[1] == 1 and row[1] == 0:
            y[idx] = 1

        # rel 5
        else:
	    y[idx] = 5

    return y


def data_chunking():
    """
    read data training
    and save them in chunks per search id
    """

    print 'Data chunking ...'

    """
    get search ids
    """

    if is_file_exist('search_ids'):
        search_ids = load_var('search_ids')
    else:
        reader = read_training_data()
        search_ids = Set([])

        for chunk in reader:
            search_ids.update(list(chunk['srch_id'].astype(int).values))

        save_var(search_ids, 'search_ids')

    """
    split the csv file
    into smaller csvs
    """

    keys = Set([])
    ids_dict = {}

    for key, rows in groupby(csv.reader(open(TRAIN_DATA)), lambda row: str(row[0])[0:2]):

        keys.update([key])

        with open(folder+"%s.csv" % key, "w") as output:

            if key == 'sr':
                columns = list(rows)[0]
            else:
                for row in rows:
                    ids_dict[row[0]] = key
                    output.write(",".join(row) + "\n")

    save_var(ids_dict, 'ids_dict')

    """
    read the smaller csv files
    and save into dataframes
    """

    keys.remove('sr')

    for key in keys:
        filename = folder+key+'.csv'
        df = pd.read_csv(filename, dtype=object, names=columns)
        save_var(df, filename)


def pointwise_preprocessing(df, columns, test=False):
    """
    simple pre-processing
    """

    # # treatment for missing values
    idx = columns['srch_query_affinity_score']
    df[(df[:,idx] == np.nan) | (df[:,idx] == 'NULL') | (df[:,idx] == '') | (df[:,idx] == 'nan'),idx] = -350

    # idx = columns['prop_location_score2']
    # df[(df[:,idx] == 'NULL') | (df[:,idx] == '') | (df[:,idx] == 'nan'),idx] = 0

    # Replace NULL of competitiors with 0 in place
    idcs = []
    idcs1 = []
    idcs2 = []
    idcs3 = []
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'

        idx1 = columns[rate]
        idx2 = columns[inv]
        idx3 = columns[diff]

        df[(df[:,idx1] == np.nan) | (df[:,idx1] == 'NULL') | (df[:,idx1] == '') | (df[:,idx1] == 'nan'),idx1] = 0
        df[(df[:,idx2] == np.nan) | (df[:,idx2] == 'NULL') | (df[:,idx2] == '') | (df[:,idx2] == 'nan'),idx2] = 0
        df[(df[:,idx3] == np.nan) | (df[:,idx3] == 'NULL') | (df[:,idx3] == '') | (df[:,idx3] == 'nan'),idx3] = 0

        idcs += [idx1, idx2, idx3]
        idcs1.append(idx1)
        idcs2.append(idx2)
        idcs3.append(idx3)

    # average over competitor data
    new_attribute1 = np.mean(df[:,idcs1].astype(float), axis=1)
    new_attribute2 = np.mean(df[:,idcs2].astype(float), axis=1)
    new_attribute3 = np.mean(df[:,idcs3].astype(float), axis=1)
    df = np.column_stack((df, new_attribute1))
    df = np.column_stack((df, new_attribute2))
    df = np.column_stack((df, new_attribute3))

    df[(df == np.nan) | (df == 'NULL') | (df == '') | (df == 'nan')] = -10

    idcs += [columns['site_id'], columns['visitor_location_country_id'], columns['prop_id'], columns['prop_country_id'], columns['srch_destination_id']]
    idcs += [columns['srch_id'], columns['date_time']]

    if test:
        X = np.delete(df,idcs,axis=1).astype(float)
        return X.astype(float)
    else:
        idcs += [
        columns['position'],
        columns['click_bool'],
        columns['gross_bookings_usd'],
        columns['booking_bool']]

        # create features matrix and target array
        X = np.delete(df,idcs,axis=1).astype(float)
        # y = pointwise_label(df[:,[columns['click_bool'],columns['booking_bool']]].astype(int))
        y = pointwise_relevance(df[:,[columns['click_bool'],columns['booking_bool']]].astype(int))

        return X, y
