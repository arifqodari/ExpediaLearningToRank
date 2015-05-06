import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import *

if __name__ == "__main__":
    # Read data
    kaggle_training_data = read_kaggle_training_data()

    indices = list()
    clicked = list()
    booked = list()

    for chunk_data in kaggle_training_data:
        # Make a unique key (the error rate is less than 0.05%)
        comb= data_chunk['date_time'].map(str)+ data_chunk['prop_country_id'].map(str)+ data_chunk['prop_id'].map(str)
        indices.append(comb)
        clicked.append(data_chunk.click_bool)
        booked.append(data_chunk.booking_bool)

    # Piece together all fragments of the data into a data frame
    indexed_kaggle_training_data = pd.concat([pd.concat(indices),pd.concat(clicked),pd.concat(booked)],axis=1)
    # Set the column names
    indexed_kaggle_training_data.columns = ['key','click_bool','booking_bool']

    # Read the test data, of the assignment, not kaggle, here I don't CHUNK the data.
    test_data = read_test_data(chunked = True)

    # Make a unique key for the test data
    comb_test= test_data['date_time'].map(str)+test_data['prop_country_id'].map(str)+ test_data['prop_id'].map(str)
    
    # Piece together the key, srch_id, and prop_id
    indexed_test_data = pd.concat([comb_test,test_data.srch_id,test_data.prop_id],axis=1)
    # Set the column names
    indexed_test_data.columns = ['key','srch_id','prop_id']

    # Do the magic merging task
    test_merge = pd.merge(indexed_test_data, indexed_kaggle_training_data,how='left')

    # Drop the key column in place to save memory
    test_merge.drop('key',axis=1, inplace = True)

    # Write it to file
    test_merge.to_csv('../data/keys_to_test.csv',sep='\t')
