import numpy as np
import pandas as pd
from setting import *
from data_reader import *
from preprocessing import *
from sklearn.datasets import *
from sklearn.externals import joblib
import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from rankpy.metrics import NormalizedDiscountedCumulativeGain
import os


# Read the test data without keys
test_data = read_test_data(chunked=False)
search_ids = test_data.srch_id.unique()
search_id_chunk = np.array_split(search_ids,20)
for num,chunk in zip(range(1,21),search_id_chunk):
    test_data[test_data.srch_id.isin(list(chunk))].to_csv('../data/test/test_data_'+str(num)+'.csv',index=False)

print 'The split task is done!'