import numpy as np
import pandas as pd

from sets import Set
from main import *


def get_ids(readers):
    """
    get visitor country id and prop country id
    """

    visitor_cid = Set([])
    prop_cid = Set([])

    for reader in readers:
        for chunk in reader:
            visitor_country_ids = chunk['visitor_location_country_id'].astype(int).values
            prop_country_ids = chunk['prop_country_id'].astype(int).values

            visitor_cid.update(np.unique(visitor_country_ids))
            prop_cid.update(np.unique(prop_country_ids))

    return visitor_cid, prop_cid


def compute_mean_std(readers, visitor_cid, prop_cid):
    """
    compute mean and std
    """

    visitor_cid = list(visitor_cid)
    prop_cid = list(prop_cid)

    # build data matrix
    sum_matrix = np.zeros((max(visitor_cid), max(prop_cid)))
    sum_squared_matrix = np.zeros((max(visitor_cid), max(prop_cid)))
    count_matrix = np.zeros((max(visitor_cid), max(prop_cid)))
    std_matrix = np.zeros((max(visitor_cid), max(prop_cid)))

    # compute mean
    for reader in readers:
        for chunk in reader:
            idx1 = chunk['visitor_location_country_id'].astype(int).values - 1
            idx2 = chunk['prop_country_id'].astype(int).values - 1
            val = chunk['orig_destination_distance'].astype(float).values

            ones = (val * 0) + np.ones(val.size)
            ones = np.nan_to_num(ones)
            val = np.nan_to_num(val)
            val2 = val**2

            sum_matrix[idx1, idx2] += val
            sum_squared_matrix[idx1, idx2] += val2
            count_matrix[idx1, idx2] += ones

    mean_matrix = sum_matrix / count_matrix
    mean_matrix = np.nan_to_num(mean_matrix)

    # compute std
    mean_squared_matrix = sum_squared_matrix / count_matrix
    mean_squared_matrix = np.nan_to_num(mean_squared_matrix)

    std_matrix = np.sqrt(mean_squared_matrix - (mean_matrix**2))

    return mean_matrix, std_matrix, sum_matrix, count_matrix


if __name__ == "__main__":
    print 'Reading data...'
    train_reader = read_training_data(10000)
    test_reader = read_test_data(10000)

    # # get visitor_country_id and prop_country_id
    # visitor_cid, prop_cid = get_ids([train_reader, test_reader])

    # # sort them
    # sorted(visitor_cid)
    # sorted(prop_cid)

    # # compute mean and std
    # print 'Computing mean and std...'
    # mean_matrix, std_matrix = compute_mean_std([train_reader, test_reader], visitor_cid, prop_cid)

    # print mean_matrix
    # print std_matrix
