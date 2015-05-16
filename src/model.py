import numpy as np
import pandas as pd
import cPickle as pickle
import csv

from setting import *
from data_reader import *
from preprocessing import *
from sklearn.ensemble import *
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm


def rf_train(X_train, y_train, n_trees=50,n_jobs=1,max_depth=None):
    """
    train using Random Forest classifier
    """

    rf0 = RandomForestClassifier(n_estimators=n_trees,
            verbose=2,
            n_jobs=n_jobs,
            max_depth=max_depth,
            class_weight='auto',
            random_state=1)

    rf1 = RandomForestClassifier(n_estimators=n_trees,
            verbose=2,
            n_jobs=n_jobs,
            max_depth=max_depth,
            class_weight='auto',
            random_state=1)

    print 'Training RF0 ...'
    rf0.fit(X_train, y_train[:,0])
    save_model(rf0, 'rf0')

    print 'Training RF1 ...'
    rf1.fit(X_train, y_train[:,1])
    save_model(rf1, 'rf1')

    return rf0, rf1


def rf_predict(X_val, y_val, rf0=None, rf1=None):
    """
    predict validation data set
    """

    print 'Predicting ...'

    if rf0 is None:
        rf0 = load_model('rf0')

    pred0 = rf0.predict_proba(X_val)[:,1]
    del rf0

    if rf1 is None:
        rf1 = load_model('rf1')
    pred1 = rf1.predict_proba(X_val)[:,1]
    del rf1


    # predict probability of each data if it is less relevant than the rel
    # classifier -- rf0 for rel1 rf1 for rel5
    # pred0 = rf0.predict_proba(X_val)[:,1]
    # pred1 = rf1.predict_proba(X_val)[:,1]

    # combine the probability
    comb_pred = np.zeros((y_val.shape[0],3))

    comb_pred[:,0] = pred0 - 0
    comb_pred[:,1] = pred1 - pred0
    comb_pred[:,2] = 1 - pred1

    # compute predicted relevance
    rel_pred = np.zeros((y_val.shape[0],2))
    rel_pred[:,0] = np.argmax(comb_pred, axis=1)
    rel_pred[:,1] = np.max(comb_pred, axis=1)
    rel_pred[rel_pred[:,0] == 2,0] = 5

    save_var(rel_pred, 'rel_pred')

    return rel_pred


def rfr_train(X_train, y_train, n_trees=50,n_jobs=1,max_depth=100):
    """
    train using Random Forest regressor
    """

    rfr = RandomForestRegressor(n_estimators=n_trees,
            verbose=2,
            n_jobs=n_jobs,
            max_depth=max_depth,
            random_state=1)

    print 'Training RFR ...'
    rfr.fit(X_train, y_train)
    save_model(rfr, 'rfr')

    return rfr


def rfr_predict(X_val, rfr=None):
    """
    predict validation data set
    """

    print 'Predicting ...'

    if rfr is None:
        rfr = load_model('rfr')

    rel_pred = rfr.predict(X_val)
    save_var(rel_pred, 'rel_pred_reg')

    return rel_pred


def rfr_predict_test(rfr=None):
    """
    predict validation data set
    """

    print 'Predicting ...'

    if rfr is None:
        rfr = load_model('rfr')

    count = 0
    rel_pred = np.ndarray(0)

    for key, rows in groupby(csv.reader(open(TEST_DATA)), lambda row: str(row[0])[0:2]):
        count += 1
        print count

        arr = np.array(list(rows))

        if key == 'sr':
            idcs = np.arange(len(arr[0]))
            columns = dict(zip(arr[0], idcs))
        else:
            X_test = pointwise_preprocessing(arr, columns, test=True)
            pred = rfr.predict(X_test)
            rel_pred = np.append(rel_pred, pred)

    save_var(rel_pred, 'rel_pred_test')

    return rel_pred
