import numpy as np
import pandas as pd
import cPickle as pickle
import copy

from sets import Set
from setting import *
from data_reader import *
from preprocessing import *
from sklearn.datasets import *


def pointwise_sampling():
    """
    Use bootstrap sampling to make sure balanced data
    """

    per_test = 0.05
    search_ids = np.array(list(load_var('search_ids')))
    n_srch_id = len(search_ids)


    print 'Sampling ...'

    n_test = int(n_srch_id * per_test)
    np.random.shuffle(search_ids)
    test_ids = search_ids[0:n_test]

    rel_val = []

    count = 0
    for key, rows in groupby(csv.reader(open(TRAIN_DATA)), lambda row: row[0]):
        count += 1
        if count % 1000 == 0:
            print count

        arr = np.array(list(rows))

        if key == 'srch_id':
            idcs = np.arange(len(arr[0]))
            columns = dict(zip(arr[0], idcs))

            idx_bool = columns['booking_bool']
            idx_click = columns['click_bool']

            arr_val = []
            arr_train = []
        elif int(key) in test_ids:
            arr_val.append(arr)
            rel_val.append([key, get_rel(arr[:,[idx_click,idx_bool]])])
        else:
            train1 = arr[np.where((arr[:,idx_click] == '1') & (arr[:,idx_bool] == '1'))]
            train2 = arr[np.where((arr[:,idx_click] == '1') & (arr[:,idx_bool] == '0'))]
            train3 = arr[np.where((arr[:,idx_click] == '0') & (arr[:,idx_bool] == '0'))]

            t1 = get_one_random_sample(train1)
            t2 = get_one_random_sample(train2)
            t3 = get_one_random_sample(train3)

            arr_train.append(t1)
            arr_train.append(t2)
            arr_train.append(t3)

    val = np.vstack(arr_val)
    train = np.vstack(arr_train)

    np.save('../data/train', train)
    np.save('../data/val', val)
    save_var(rel_val, 'rel_val')
    save_var(columns, 'columns')

    return train, val, rel_val, columns


def get_one_random_sample(X):
    if X.shape[0] > 1:
        random = np.random.randint(X.shape[0])
        X = X[random,:]

    return X


def get_rel(arr):
    rel = np.sum(arr.astype(int), axis=1)
    rel[rel == 2] = 5
    return rel


def extract_test_rel():

    rel_test = []

    count = 0
    for key, rows in groupby(csv.reader(open('../data/test_only_with_keys.csv','r')), lambda row: row[0]):
        count += 1
        if count % 1000 == 0:
            print count

        arr = np.array(list(rows))

        if key == 'srch_id':
            idcs = np.arange(len(arr[0]))
            columns = dict(zip(arr[0], idcs))

            idx_bool = columns['booking_bool']
            idx_click = columns['click_bool']
        else:
            rel_test.append([key, get_rel(arr[:,[idx_click,idx_bool]])])

    save_var(rel_test, 'rel_test')

    return rel_test


def listwise_sampling():

    per_test = 0.05
    search_ids = np.array(list(load_var('search_ids')))
    n_srch_id = len(search_ids)


    print 'Sampling ...'

    n_train = 500000
    n_test = int(n_srch_id * per_test)
    np.random.shuffle(search_ids)
    test_ids = search_ids[0:n_test]

    output_train = file('../data/svmlight_train.txt', 'a')
    output_val = file('../data/svmlight_val.txt', 'a')

    count = -1
    count_row = 0
    for key, rows in groupby(csv.reader(open(TRAIN_DATA)), lambda row: row[0]):
        count += 1
        if count % 1000 == 0:
            print count

        arr = np.array(list(rows))

        if key == 'srch_id':
            idcs = np.arange(len(arr[0]))
            columns = dict(zip(arr[0], idcs))

            idx_bool = columns['booking_bool']
            idx_click = columns['click_bool']
        elif int(key) in test_ids:
            rel_val = get_rel(arr[:,[idx_click,idx_bool]])
            X_val, y_val = pointwise_preprocessing(arr, columns)
            dump_svmlight_file(X_val, rel_val, output_val, query_id=arr[:,0].astype(int))
        else:
            if count_row < n_train:
                rel_train = get_rel(arr[:,[idx_click,idx_bool]])
                X_train, y_train = pointwise_preprocessing(arr, columns)
                dump_svmlight_file(X_train, rel_train, output_train, query_id=arr[:,0].astype(int))
                count_row += arr.shape[0]


def listwise_sampling_test():
    count = -1
    test_srch_ids = np.ndarray(0)
    output = file('../data/svmlight_test.txt', 'a')
    rels = load_var('rel_test')

    for key, rows in groupby(csv.reader(open(TEST_DATA)), lambda row: row[0]):

        arr = np.array(list(rows))

        if key == 'srch_id':
            idcs = np.arange(len(arr[0]))
            test_columns = dict(zip(arr[0], idcs))
        else:
            X_test = pointwise_preprocessing(arr, test_columns, test=True)
            dump_svmlight_file(X_test, rels[count][1], output, query_id=arr[:,0].astype(int))

        count += 1
        if count % 1000 == 0:
            print count
