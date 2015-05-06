import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import *

if __name__ == "__main__":
    training_reader = read_training_data()

    for data_chunk in training_reader:

        #################### NULL Corection ####################
        # Remove srch_id and date_time
        data_chunk.drop(['srch_id','date_time'],axis = 1, inplace = True)

        # Replace NULL with -10 in place
        data_chunk.visitor_hist_starrating.fillna(-10,inplace = True)

        data_chunk.visitor_hist_adr_usd.fillna(-10,inplace = True)

        data_chunk.prop_review_score.fillna(-10, inplace = True)

        # Replace NULL of competitiors with 0 in place
        for i in range(1,9):
            rate = 'comp' + str(i) + '_rate'
            inv = 'comp' + str(i) + '_inv'
            diff = 'comp' + str(i) + '_rate_percent_diff'
            data_chunk[rate].fillna(0,inplace = True)
            data_chunk[inv].fillna(0,inplace = True)
            data_chunk[diff].fillna(0,inplace = True)

		############### Feature Transformation #################
