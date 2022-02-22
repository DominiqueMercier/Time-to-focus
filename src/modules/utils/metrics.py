import numpy as np
from dtaidistance import dtw_ndim
from scipy.stats import pearsonr, spearmanr


def pearsonr_func(a, b):
    return pearsonr(a, b)[0]


def spearmanr_func(a, b):
    return spearmanr(a, b)[0]


def jaccard_func(a, b):
    set_a, set_b = set(a), set(b)
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def compute_compare(data, dataB=None, mode='pearsonr'):
    if mode == 'spearmanr':
        func = spearmanr_func
    elif mode == 'dtw':
        func = dtw_ndim.distance
    elif mode == 'jaccard':
        func = jaccard_func
    else:
        func = pearsonr_func
    dataB = data if dataB is None else dataB
    mat = np.ones((data.shape[0], dataB.shape[0]))
    for i in range(data.shape[0]):
        for j in range(dataB.shape[0]):
            mat[i, j] = func(data[i], dataB[j])
    return mat


def compute_continuity(data):
    con = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        con[i] = np.mean(np.absolute(data[i, :, 1:] - data[i, :, :-1]))
    return con
