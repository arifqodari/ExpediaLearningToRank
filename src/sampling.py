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
            train = np.ndarray((0,idcs.size))

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
