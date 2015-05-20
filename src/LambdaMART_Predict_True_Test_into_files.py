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


# In[5]:

## Global parameters
avg_numerics = pickle.load(open('../data/numeric_per_prop_id_avg_mean_std_competitors.pkl'))
model = joblib.load('../modellers/model_012/model_012.pkl')
LOG = model_log_folder + 'prediction_model12.log'


# In[12]:

# A relevance function to define the relevance score for NDCG
def relevance(a):
    if a[0] == a[1] == 1:
        return 5
    elif a[0] == 1 and a[1] == 0:
        return 1
    else:
        return 0

# Data processing on true test data
def data_preprocess(test_data):
    preprocessing_3(test_data,type = 0)
    # Add extra features to true test data
    test_data_new = pd.merge(test_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
    test_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
    col_names = list(test_data_new.columns)
    col_names.remove('click_bool')
    col_names.remove('booking_bool')
    col_names.remove('srch_id')
    dump_svmlight_file(test_data_new[col_names].values,test_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_true_test_chunk_temp1.txt',query_id = test_data_new.srch_id)
    print "Data preprocessing is done and SVMlight format file is generated..."


# In[16]:

def output_prediction(test_data,head=True):
    test_queries = Queries.load_from_text('../data/svmlight_true_test_chunk_temp1.txt',purge=None)
    metric = NormalizedDiscountedCumulativeGain(38, queries=test_queries)
    # Make the prediction
    predict_scores = model.predict(test_queries, n_jobs=-1)
    # Extract the srch_id and prop_id into a dataframe
    result_unordered = test_data.loc[:,['srch_id','prop_id']]
    result_unordered['scores'] = predict_scores
    result_ordered = result_unordered.sort(['srch_id','scores'],ascending=[1,0])
    # Write the submission into file
    result_ordered.loc[:,['srch_id','prop_id']].to_csv('../data/predict_test_data.csv',index=False, mode='a',header = head)
    print "Prediction has been written inte the file..."


# In[17]:

first = False
for i in range(16,17):
    test_data_temp = pd.read_csv('../data/test_with_key/test_data_'+ str(i) +'.csv')
    data_preprocess(test_data_temp)
    output_prediction(test_data_temp,first)
    #first = False

