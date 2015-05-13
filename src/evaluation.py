import numpy as np
import pandas as pd
import cPickle as pickle

from setting import *
from data_reader import *


def eval_ndcg(pred, rel_val):
    """
    compute ndcg at 38 of prediction
    """

    print 'Evaluating mean NDCG ...'

    i_pred = 0
    mean_ndcg = 0

    for r in rel_val:

        dic = r[1]
        rels = sorted(r[1], reverse=True)
        n_rels = len(rels)

        sub_pred = pred[i_pred:i_pred+n_rels,:]
        rank_pred = np.lexsort((-sub_pred[:,1], -sub_pred[:,0]))

        i = 0
        dcg = 0
        idcg = 0

        while i < 38 and i < len(rels):

            inumerator = 2**rels[i] - 1
            idenominator = np.log2(i + 2)
            idcg += float(inumerator / idenominator)

            numerator = 2**dic[rank_pred[i]] - 1
            denominator = np.log2(i + 2)
            dcg += float(numerator / denominator)

            i += 1
            i_pred += 1

        mean_ndcg += dcg / idcg

    return mean_ndcg / len(rel_val)


def eval_ndcg_reg(pred, rel_val):
    """
    compute ndcg at 38 of prediction
    """

    print 'Evaluating mean NDCG ...'

    i_pred = 0
    mean_ndcg = 0

    for r in rel_val:

        dic = r[1]
        rels = sorted(r[1], reverse=True)
        n_rels = len(rels)

        sub_pred = pred[i_pred:i_pred+n_rels]
        rank_pred = np.argsort(-sub_pred)

        i = 0
        dcg = 0
        idcg = 0

        while i < 38 and i < len(rels):

            inumerator = 2**rels[i] - 1
            idenominator = np.log2(i + 2)
            idcg += float(inumerator / idenominator)

            numerator = 2**dic[rank_pred[i]] - 1
            denominator = np.log2(i + 2)
            dcg += float(numerator / denominator)

            i += 1
            i_pred += 1

        mean_ndcg += dcg / idcg

    return mean_ndcg / len(rel_val)
