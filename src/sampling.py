import numpy as np
import pandas as pd
import cPickle as pickle
import copy

from sets import Set
from setting import *
from data_reader import *
from preprocessing import *


def pointwise_sampling():
    """
    Use bootstrap sampling to make sure balanced data
    """

    per_test = 0.05
    search_ids = np.array(list(load_var('search_ids')))
    # ids_dict = load_var('ids_dict')
    n_srch_id = len(search_ids)


    """
    sample randomly 5% of search ids (including their results) for tests
    """

    print 'Sampling ...'

    n_test = int(n_srch_id * per_test)
    np.random.shuffle(search_ids)
    test_ids = search_ids[0:n_test]

    save_var(test_ids, 'test_ids')
    # first = True

    # for test_id in test_ids:

    #     filename = ids_dict[str(test_id)]+'.csv'
    #     rawdf = load_var(filename)
    #     df = rawdf[rawdf['srch_id'] == test_id]
    #     X, y = pointwise_preprocessing(df)

    #     if first:
    #         X_val = X
    #         y_val = y
    #         first = False
    #     else:
    #         X_val = np.vstack((X_val, X))
    #         y_val = np.concatenate((y_val, y))

    # print len(test_ids)
    # print X_test.shape

    # df_val = pd.DataFrame()
    first_val = True
    first_train = True

    rel_val = []

    count = 0
    for key, rows in groupby(csv.reader(open(TRAIN_DATA)), lambda row: row[0]):
        count += 1
        if count % 1000 == 0:
            print count

        arr = np.array(list(rows))
        arr[(arr == 'NULL') | (arr == '') | (arr == 'nan')] = np.nan

        if key == 'srch_id':
            cols = list(arr[0])
            idcs = np.arange(len(cols))
            columns = dict(zip(cols, idcs))

            idx_bool = columns['booking_bool']
            idx_click = columns['click_bool']
        elif int(key) in test_ids:
            # df_val = df_val.append(pd.DataFrame(arr, columns=columns))
            if first_val:
                val = arr
                first_val = False
            else:
                val = np.vstack((val, arr))

            rel_val.append([key, get_rel(arr[:,[idx_click,idx_bool]])])
        else:
            # df_train = pd.DataFrame(arr, columns=columns)

            train1 = arr[(arr[:,idx_click] == '1') & (arr[:,idx_bool] == '1')]
            train2 = arr[(arr[:,idx_click] == '1') & (arr[:,idx_bool] == '0')]
            train3 = arr[(arr[:,idx_click] == '0') & (arr[:,idx_bool] == '0')]

            train1 = get_one_random_sample(train1)
            train2 = get_one_random_sample(train2)
            train3 = get_one_random_sample(train3)

            if first_train:
                train = np.vstack((train1, train2, train3))
                first_train = False
            else:
                train = np.vstack((train, train1, train2, train3))

            # get book sample
            # X1, y1 = get_book_sample(df_train)

            # get click sample
            # X2, y2 = get_click_sample(df_train)

            # get the irrelevant sample
            # X3, y3 = get_irrelevant_sample(df_train)

            # if first:
            #     X_train = np.vstack((X1, X2, X3))
            #     y_train = np.append(np.append(y1,y2), y3)
            #     first = False
            # else:
            #     X_train = np.vstack((X_train, X1, X2, X3))
            #     y_train = np.append(np.append(np.append(y_train, y1), y2), y3)

        if count % 10000 == 0:
            X_val, y_val = pointwise_preprocessing(val, columns)
            X_train, y_train = pointwise_preprocessing(train, columns)
            save_var([X_train, y_train, X_val, y_val, rel_val], 'xy')


    # df_val = pd.DataFrame(val, columns=columns)
    # df_train = pd.DataFrame(train, columns=columns)

    X_val, y_val = pointwise_preprocessing(val, columns)
    X_train, y_train = pointwise_preprocessing(train, columns)
    # print X_val
    # print len(test_ids)
    # print X_val.shape

    # print len(search_ids) - len(test_ids)
    # print X_train.shape
    # print y_train.shape
    # print rel_val


    """
    the rest search ids will be used as training
    sampling for training test
    for each srch_id sample one book, one click, one irrelevant
    to make sure we have balanced data for all classes
    """

    # print 'Sampling training set ...'

    # train_ids = search_ids[n_test:]
    # first = True

    # for train_id in train_ids:

    #     filename = ids_dict[str(train_id)]+'.csv'
    #     rawdf = load_var(filename)
    #     df = rawdf[rawdf['srch_id'] == train_id]

    #     # get book sample
    #     X1, y1 = get_book_sample(df)

    #     # get click sample
    #     X2, y2 = get_click_sample(df)

    #     # get the irrelevant sample
    #     X3, y3 = get_irrelevant_sample(df)

    #     if first:
    #         X_train = np.vstack((X1, X2, X3))
    #         y_train = np.append(np.append(y1,y2), y3)
    #         first = False
    #     else:
    #         X_train = np.vstack((X_train, X1, X2, X3))
    #         y_train = np.append(np.append(np.append(y_train, y1), y2), y3)

    # print len(train_ids)
    # print X_train.shape
    # print y_train.shape

    # save train and val set
    save_var([X_train, y_train, X_val, y_val, rel_val], 'xy')

    return X_train, y_train, X_val, y_val, rel_val


def get_one_random(df):
    X_temp, y_temp = pointwise_preprocessing(df)

    if X_temp.shape[0] > 1:
        n_row = X_temp.shape[0]
        random = np.random.randint(n_row)
        X = X_temp[random,:]
        y = y_temp[random]

        return X, y
    else:
        return X_temp, y_temp


def get_book_sample(chunk):
    """
    get one sample of booked hotel
    """

    df = chunk[(chunk['booking_bool'].astype(int) == 1) & (chunk['click_bool'].astype(int) == 1)]
    return get_one_random(df)


def get_click_sample(chunk):
    """
    get one sample of clicked hotel
    """

    df = chunk[(chunk['booking_bool'].astype(int) == 0) & (chunk['click_bool'].astype(int) == 1)]
    return get_one_random(df)


def get_irrelevant_sample(chunk):
    """
    get one sample of irrelevant hotel
    """

    df = chunk[(chunk['booking_bool'].astype(int) == 0) & (chunk['click_bool'].astype(int) == 0)]
    return get_one_random(df)


def get_one_random_sample(X):
    if X.shape[0] > 1:
        n_row = X.shape[0]
        random = np.random.randint(n_row)
        X = X[random,:]

    return X


def get_rel(arr):
    rel = np.sum(arr.astype(int), axis=1)
    rel[rel == 2] = 5
    return rel
