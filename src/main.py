import numpy as np
import pandas as pd
import cPickle as pickle
import sys

from sets import Set
from setting import *
from data_reader import *
from preprocessing import *
from sampling import *
from model import *
from evaluation import *
from sklearn.ensemble import *




if __name__ == "__main__":
    """
    training
    """
    # train, val, rel_val, columns = pointwise_sampling()

    train = np.load('../data/train.npy')
    val = np.load('../data/val.npy')
    columns = load_var('columns')
    rel_val = load_var('rel_val')

    X_train, y_train = pointwise_preprocessing(train, columns)
    X_val, y_val = pointwise_preprocessing(val, columns)
    del train, val, columns

    # rf0, rf1 = rf_train(X_train, y_train, n_trees=100, n_jobs=-1, max_depth=10)
    # pred = rf_predict(X_val, y_val, rf0, rf1)
    # ndcg = eval_ndcg(pred, rel_val)

    rfr = rfr_train(X_train, y_train, n_trees=400, n_jobs=-1, max_depth=10)
    pred = rfr_predict(X_val, rfr)
    ndcg = eval_ndcg_reg(pred, rel_val)

    print ndcg


    """
    testing
    """
    # rel_test = extract_test_rel()
    # rel_test = load_var('rel_test')
    # pred = rfr_predict_test()

    # ndcg = eval_ndcg_reg(pred, rel_test)
    # print ndcg
