import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def scale_data(trainX, testX, mode='standardize', frange=(0, 1)):
    scaler = StandardScaler() if mode == 'standardize' else MinMaxScaler(feature_range=frange)
    org_train_shape = trainX.shape[1:]
    org_test_shape = testX.shape[1:]
    trainXf = trainX.reshape(-1, org_train_shape[-1])
    testXf = testX.reshape(-1, org_test_shape[-1])
    scaler.fit(trainXf)
    trainX = scaler.transform(trainXf).reshape(-1, *org_train_shape)
    testX = scaler.transform(testXf).reshape(-1, *org_test_shape)
    return trainX, testX


def preprocess_data(trainX, trainY, testX, testY, normalize=False, standardize=True, frange=(0, 1), channel_first=True):
    # adjust labels
    le = LabelEncoder().fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    # remove missings
    cmean = np.nanmean(trainX, axis=(0, 1))
    inds = np.where(np.isnan(trainX))
    trainX[inds] = np.take(cmean, inds[2])
    inds = np.where(np.isnan(testX))
    testX[inds] = np.take(cmean, inds[2])
    # standardize
    if standardize:
        trainX, testX = scale_data(trainX, testX, mode='standardize')
    # normalize
    if normalize:
        trainX, testX = scale_data(
            trainX, testX, mode='normalize', frange=frange)
    if channel_first:
        trainX = trainX.transpose(0, 2, 1)
        testX = testX.transpose(0, 2, 1)
    return trainX, trainY, testX, testY


def perform_datasplit(data, labels, test_split=0.3, stratify=True, return_state=False, random_state=0):
    try:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state, stratify=labels if stratify else None)
        state = True
    except:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state, stratify=None)
        print('Warining: No stratified split possible')
        state = False
    if return_state:
        return da, la, db, lb, state
    return da, la, db, lb


def fuse_train_val(trainX, trainY, valX, valY):
    split_id = trainY.shape[0]
    trainX, trainY = np.stack([trainX, valX]), np.stack([trainY, valY])
    return trainX, trainY, split_id


def unfuse_train_val(data, labels, split_id):
    trainX, trainY = data[:split_id], labels[:split_id]
    valX, valY = data[split_id:], labels[split_id:]
    return trainX, trainY, valX, valY


def select_subset(data, labels, thresh=100, at_least=1, stratified=True):
    _, cwise_counts = np.unique(labels, return_counts=True)
    cwise_ids = [[] for _ in cwise_counts]
    for i, l in enumerate(labels):
        cwise_ids[l].append(i)
    if thresh < 1:
        thresh = labels.shape[0] * thresh
    if stratified:
        keep = [c*thresh/np.sum(cwise_counts) for c in cwise_counts]
    else:
        keep = [100/len(cwise_counts) for _ in cwise_counts]
    keep = np.array([np.round(v) for v in keep], dtype=int)
    keep = [np.max([at_least, k]) for k in keep]
    keep_ids = np.concatenate([cwise_ids[i][:keep[i]]
                              for i in range(len(keep))])
    sub_data = data[keep_ids]
    sub_labels = labels[keep_ids]
    return sub_data, sub_labels, keep_ids


def randomize_labels(labels):
    return np.random.permutation(labels)
