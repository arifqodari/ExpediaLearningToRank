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
    X_train, y_train, X_val, y_val, rel_val = pointwise_sampling()
    # X_train, y_train, X_val, y_val, rel_val = load_var('xy')
    rf0, rf1 = rf_train(X_train, y_train, n_jobs=2)
    pred = rf_predict(X_val, y_val, rf0, rf1)
    # pred = rf_predict(X_val, y_val)

    pred = load_var('rel_pred')
    ndcg = eval_ndcg(pred, rel_val)
    print ndcg
