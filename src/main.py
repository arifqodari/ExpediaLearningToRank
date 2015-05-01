import numpy as np
import pandas as pd

from data import *


def read_data():
    """
    we don't have enough memoery to load a big file
    so we need to read it in chunks
    """

    return pd.read_csv(TEST_DATA, dtype=object, chunksize=30)


if __name__ == "__main__":
    reader = read_data()

    for chunk in reader:
        print chunk
