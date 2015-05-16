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

# Read the data into memory
training_data = pd.read_csv(TRAIN_DATA,nrows= 2500000)
validation_data = pd.read_csv(TRAIN_DATA,skiprows=2500000,nrows = 1000000, header=False, names= training_data.columns)
test_data = pd.read_csv(TRAIN_DATA,skiprows= 3500000,nrows = 1000000, header=False,names= training_data.columns)

# Fist step of data preprocessing
preprocessing_3(training_data)
preprocessing_3(validation_data)
preprocessing_3(test_data)

# Read the attribute dictionary (numeric attributes per prop_id)
avg_numerics = pickle.load(open('../data/numeric_per_prop_id_avg_mean_std_competitors.pkl'))

# Add the new attributes to the original data
training_data_new = pd.merge(training_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
validation_data_new = pd.merge(validation_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
test_data_new = pd.merge(test_data,avg_numerics,how='left',left_on='prop_id',right_on='prop_id_',sort=False)
training_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
validation_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)
test_data_new.drop(['prop_id','prop_id_'],axis=1,inplace=True)

col_names = list(training_data_new.columns)
col_names.remove('click_bool')
col_names.remove('booking_bool')
col_names.remove('srch_id')

# A relevance function to define the relevance score for NDCG
def relevance(a):
    if a[0] == a[1] == 1:
        return 5
    elif a[0] == 1 and a[1] == 0:
        return 1
    else:
        return 0

# Generate the SVMLight format file
dump_svmlight_file(training_data_new[col_names].values,training_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_training_avg_mean_std_competitors_m2.txt',query_id=training_data_new.srch_id)
dump_svmlight_file(validation_data_new[col_names].values,validation_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_validation_avg_mean_std_competitors_m2.txt',query_id=validation_data_new.srch_id)
dump_svmlight_file(test_data_new[col_names].values,test_data.iloc[:,-2:].apply(relevance,axis = 1),'../data/svmlight_test_avg_mean_std_competitors_m2.txt',query_id = test_data_new.srch_id)

# Please set it for each training
model_name = 'model_012'
# Parameters for file recording
LOG = model_log_folder + model_name +'.log'
MODELLER_DIR = modeller_folder + model_name + '/'

# Turn on logging.
logging.basicConfig(filename = LOG,format='%(asctime)s : %(message)s', level=logging.INFO)

# Load the query datasets.
train_queries = Queries.load_from_text('../data/svmlight_training_avg_mean_std_competitors_m2.txt')
pickle_output_train = open('../data/train_queries2.pkl','w')
pickle.dump(train_queries,pickle_output_train)
pickle_output_train.close()
valid_queries = Queries.load_from_text('../data/svmlight_validation_avg_mean_std_competitors_m2.txt')
pickle_output_valid = open('../data/valid_queries2.pkl','w')
pickle.dump(valid_queries,pickle_output_valid)
pickle_output_valid.close()
test_queries = Queries.load_from_text('../data/svmlight_test_avg_mean_std_competitors_m2.txt')
pickle_output_test = open('../data/test_queries2.pkl','w')
pickle.dump(test_queries,pickle_output_test)
pickle_output_test.close()

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
model = LambdaMART(n_estimators=10000, max_depth=2, shrinkage=0.07, estopping=100, n_iterations= 3, n_jobs=-1)
model.fit(metric, train_queries, validation=valid_queries)

logging.info('================================================================================')
# Save the model to files
os.mkdir(MODELLER_DIR)
logging.info('New folder is created: %s' % MODELLER_DIR)
joblib.dump(model, MODELLER_DIR + model_name + '.pkl') 
logging.info('Model is saves as: %s' % MODELLER_DIR + model_name + '.pkl')

logging.info('================================================================================')

# Print out the performance on the test set.
logging.info('%s on the test queries: %.8f' % (metric, metric.evaluate_queries(test_queries, model.predict(test_queries, n_jobs=-1))))