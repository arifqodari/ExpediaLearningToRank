
# coding: utf-8

# In[1]:

cd E:\Data Mining\ExpediaLearningToRank\src


# In[33]:

import numpy as np
import pandas as pd
from setting import *
from data_reader import *
from preprocessing import *
from sklearn.datasets import *
import logging
from rankpy.queries import Queries
from rankpy.models import LambdaMART
from rankpy.metrics import NormalizedDiscountedCumulativeGain


# In[34]:

training_data = pd.read_csv(TRAIN_DATA,nrows= 10000)
validation_data = pd.read_csv(TRAIN_DATA,skiprows=20000,nrows = 10000, header=False, names= training_data.columns)
test_data = pd.read_csv(TRAIN_DATA,skiprows= 100000,nrows = 20000, header=False,names= training_data.columns)


# In[35]:

preprocessing_1(training_data)
preprocessing_1(validation_data)
preprocessing_1(test_data)


# In[48]:

# A relevance function to define the relevance score for NDCG
def relevance(a):
    if a[0] == a[1] == 1:
        return 5
    elif a[0] == 1 and a[1] == 0:
        return 1
    else:
        return 0
#Use it here: training_data.iloc[:,-2:].apply(relevance,axis = 1)


# In[50]:

dump_svmlight_file(training_data.iloc[:,1:-2].values,training_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_training.txt',query_id=training_data.srch_id)


# In[51]:

dump_svmlight_file(validation_data.iloc[:,1:-2].values,validation_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_validation.txt',query_id=validation_data.srch_id)


# In[52]:

dump_svmlight_file(test_data.iloc[:,1:-2].values,test_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_test.txt',query_id = test_data.srch_id)


# In[54]:

# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

# Load the query datasets.
train_queries = Queries.load_from_text('../data/svmlight_training.txt')
valid_queries = Queries.load_from_text('../data/svmlight_validation.txt')
test_queries = Queries.load_from_text('../data/svmlight_test.txt')

logging.info('================================================================================')

# Save them to binary format ...
#train_queries.save('../data/train_bin')
#valid_queries.save('../data/validation_bin')
#test_queries.save('../data/test_bin')

# ... because loading them will be then faster.
#train_queries = Queries.load('../data/train_bin')
#valid_queries = Queries.load('../data/validation_bin')
#test_queries = Queries.load('../data/test_bin')

logging.info('================================================================================')

# Print basic info about query datasets.
logging.info('Train queries: %s' % train_queries)
logging.info('Valid queries: %s' % valid_queries)
logging.info('Test queries: %s' %test_queries)

logging.info('================================================================================')

# Prepare metric for this set of queries.
metric = NormalizedDiscountedCumulativeGain(38, queries=[train_queries, valid_queries, test_queries])

# Initialize LambdaMART model and train it.
model = LambdaMART(n_estimators=10000, max_depth=4, shrinkage=0.08, estopping=100, n_jobs=-1)
model.fit(metric, train_queries, validation=valid_queries)

logging.info('================================================================================')

# Print out the performance on the test set.
logging.info('%s on the test queries: %.8f' % (metric, metric.evaluate_queries(test_queries, model.predict(test_queries, n_jobs=-1))))

