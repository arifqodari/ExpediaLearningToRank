import numpy as np
import pandas as pd
from main import *

if __name__ == "__main__":
    # Read data
    kaggle_training_data = read_kaggle_training_data()

    indices = list()
    clicked = list()
    booked = list()

    for chunk_data in kaggle_training_data:
        # Make a unique key (the error rate is 0)
        comb= data_chunk['date_time'].map(str)+ data_chunk['visitor_location_country_id'].map(str) + data_chunk['prop_country_id'].map(str)+ data_chunk['prop_id'].map(str) + data_chunk['srch_adults_count'].map(str) + data_chunk['srch_children_count'].map(str) + data_chunk['srch_room_count'].map(str)+ data_chunk['prop_location_score2'].map(str)+ data_chunk['orig_destination_distance'].map(str)
        indices.append(comb)
        clicked.append(data_chunk.click_bool)
        booked.append(data_chunk.booking_bool)

    # Piece together all fragments of the data into a data frame
    indexed_kaggle_training_data = pd.concat([pd.concat(indices),pd.concat(clicked),pd.concat(booked)],axis=1)
    # Set the column names
    indexed_kaggle_training_data.columns = ['key','click_bool','booking_bool']

    # Read the test data, of the assignment, not kaggle, here I don't CHUNK the data.
    test_data = read_test_data()

    # Make a unique key for the test data
    comb_test= test_data['date_time'].map(str)+ test_data['visitor_location_country_id'].map(str) + test_data['prop_country_id'].map(str)+ test_data['prop_id'].map(str) + test_data['srch_adults_count'].map(str) + test_data['srch_children_count'].map(str) + test_data['srch_room_count'].map(str) + test_data['prop_location_score2'].map(str) + test_data['orig_destination_distance'].map(str)
    
    # Add a key column in test_data
    test_data['key']= comb_test

    test_merge = pd.merge(test_data, indexed_kaggle_training_data,on='key',how='left')
    test_merge.drop('key',axis=1, inplace = True)

    test_merge.to_csv('../data/test_with_keys.csv',index=False)

    # Piece together the key, srch_id, and prop_id
    indexed_test_data = pd.concat([comb_test,test_data.srch_id,test_data.prop_id],axis=1)
    # Set the column names
    indexed_test_data.columns = ['key','srch_id','prop_id']

    # Do the magic merging task
    indexed_test_merge = pd.merge(indexed_test_data, indexed_kaggle_training_data,how='left')

    # Drop the key column in place to save memory
    indexed_test_merge.drop('key',axis=1, inplace = True)

    # Write it to file
    indexed_test_merge.to_csv('../data/test_only_with_keys.csv',index=False)