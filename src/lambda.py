import numpy as np
import pandas as pd
from sampling import *
from setting import *
from data_reader import *
from preprocessing import *
from sklearn.datasets import *
import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from rankpy.metrics import NormalizedDiscountedCumulativeGain



logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
logging.info('dump train file')

# listwise_sampling()

logging.info('================================================================================')
logging.info('dump test file')

# listwise_sampling_test()

logging.info('================================================================================')
logging.info('load query database')

train_queries = Queries.load_from_text('../data/svmlight_train.txt')
valid_queries = Queries.load_from_text('../data/svmlight_val.txt')
test_queries = Queries.load_from_text('../data/svmlight_test.txt')

logging.info('================================================================================')
logging.info('train LambdaMart')

# Prepare metric for this set of queries.
metric = NormalizedDiscountedCumulativeGain(38, queries=[train_queries, valid_queries, test_queries])

# Initialize LambdaMART model and train it.
model = LambdaMART(n_estimators=10000, max_depth=5, shrinkage=0.08, estopping=100, n_jobs=-1, n_iterations=100)
model.fit(metric, train_queries, validation=valid_queries)

logging.info('================================================================================')
logging.info('test')

# Print out the performance on the test set.
logging.info('%s on the test queries: %.8f' % (metric, metric.evaluate_queries(test_queries, model.predict(test_queries, n_jobs=-1))))
