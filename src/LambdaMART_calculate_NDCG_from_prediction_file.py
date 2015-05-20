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
import time

# In[3]:

# The test file with answers
answers = pd.read_csv('../data/test_only_with_keys.csv')


# In[4]:

# Prediction ile
predicts = pd.read_csv('../prediction/predict_test_model12_complete.csv')
predicts.columns = ['srch_id','prop_id']


# In[7]:

# Genereate the information dataframe
documents = answers.loc[:,['srch_id','prop_id']]
documents['rel'] = answers.click_bool + 4 * answers.booking_bool
documents = pd.merge(predicts,documents,on=['srch_id','prop_id'],how='left')


# In[12]:

# Calculate the dcg score
def dcg(rel):
    numerator =2**rel - 1
    denominator = np.log2(np.arange(2,len(rel)+2))
    return (numerator/denominator).sum()


start_time = time.clock()
score_list = [dcg(group.rel)/dcg(group.rel.sort(ascending=False,inplace=False)) for name,group in documents.groupby('srch_id')]
ndcg_score = np.mean(score_list)
print time.clock() - start_time
print ndcg_score

