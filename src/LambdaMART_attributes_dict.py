import numpy as np
import pandas as pd
from setting import *
from data_reader import *
from preprocessing import *
import os


# In[3]:

training_data = read_training_data(chunked=False)


# In[4]:

test_data = read_test_data(chunked=False)


# In[5]:

preprocessing_3(training_data)
preprocessing_3(test_data,0)


# In[6]:

combination = pd.concat([training_data.iloc[:,:-2],test_data],axis=0)


# In[7]:

combination.drop('srch_id',axis=1,inplace=True)


# In[8]:

combination_groupby = combination.groupby('prop_id',sort=True).agg([np.median, np.mean, np.std])


# In[9]:

combination_groupby_reset_index = combination_groupby.reset_index()


# In[10]:

combination_groupby_reset_index.columns = ['_'.join(col).strip() for col in combination_groupby_reset_index.columns.values]


# In[19]:

combination_groupby_reset_index.fillna(0,inplace=True)


# In[20]:

pickle_output = open('../data/numeric_per_prop_id_avg_mean_std_competitors.pkl','w')
pickle.dump(combination_groupby_reset_index,pickle_output)
pickle_output.close()

