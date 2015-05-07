import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from data_reader import *
from preprocessing import *
from sampling import *
from model import *
from evaluation import *
from sklearn.ensemble import *




if __name__ == "__main__":
    train_reader = read_training_data()
    test_reader = read_test_data()

    df_train = train_reader.get_chunk(1000)
    df_test = test_reader.get_chunk(1000)

    preprocessing(df_train)

    clf = RandomForestClassifier(n_estimators=50,
            verbose=2,
            n_jobs=2,
            min_samples_split=10,
            random_state=1)
    #clf.fit(X_train, y_train)

