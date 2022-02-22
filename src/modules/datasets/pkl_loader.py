import pickle


def load_data(path, is_channel_first=False):
    with open(path, 'rb') as f:
        content = pickle.load(f)
    # check if validation set exists
    if len(content) == 4:
        trainX, trainY, testX, testY = content
        valX, valY = None, None
    else:
        trainX, trainY, valX, valY, testX, testY = content
    # correct shape
    if len(trainX.shape) == 2:
        trainX = trainX.reshape(*trainX.shape, 1)
        trainY = trainX.reshape(*trainY.shape, 1)
    elif is_channel_first:
        trainX = trainX.transpose(0, 2, 1)
        testX = testX.transpose(0, 2, 1)
    return trainX, trainY, valX, valY, testX, testY
