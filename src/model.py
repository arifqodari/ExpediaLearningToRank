import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from data_reader import *
from sklearn.ensemble import *


def rf_train(X_train, y_train, n_trees=50,n_jobs=1,min_samples_split=10):
    """
    train using Random Forest classifier
    """

    rf0 = RandomForestClassifier(n_estimators=n_trees,
            verbose=2,
            n_jobs=n_jobs,
            min_samples_split=min_samples_split,
            class_weight='auto',
            random_state=1)

    rf1 = RandomForestClassifier(n_estimators=n_trees,
            verbose=2,
            n_jobs=n_jobs,
            min_samples_split=min_samples_split,
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

    if rf0 is None or rf1 is None:
        rf0 = load_model('rf0')
        rf1 = load_model('rf1')

    # predict probability of each data if it is less relevant than the rel
    # classifier -- rf0 for rel1 rf1 for rel5
    pred0 = rf0.predict_proba(X_val)[:,1]
    pred1 = rf1.predict_proba(X_val)[:,1]

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
