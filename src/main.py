import numpy as np
import pandas as pd
import cPickle as pickle
import sys

from setting import *
from data_reader import *
from preprocessing import *
from sampling import *
from model import *
from evaluation import *
from sklearn.ensemble import *




if __name__ == "__main__":
    # train, val, rel_val, columns= pointwise_sampling()

    train = np.load('../data/train.npy')
    val = np.load('../data/val.npy')

    rel_val = load_var('rel_val')
    columns = load_var('columns')

    X_val, y_val = pointwise_preprocessing(val, columns)
    X_train, y_train = pointwise_preprocessing(train, columns)

    # rf0, rf1 = rf_train(X_train, y_train, n_trees=100, n_jobs=2)
    # pred = rf_predict(X_val, y_val, rf0, rf1)

    # ndcg = eval_ndcg(pred, rel_val)
    # print ndcg
