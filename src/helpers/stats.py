import numpy as np


def row_zscore(a):
    mean = np.mean(a, axis=1)
    std = np.std(a, axis=1)
