import torch


def normalize(tensor, eps=1.0e-7):
    tensor -= tensor.min()
    tensor /= (tensor.max() + eps)
    return tensor


def extract_subtensor(tensor: torch.Tensor, ids_time, ids_feature):
    """
        This method extracts a subtensor specified with the indices
        :param tensor: the (T, N_features) tensor from which the data should be extracted
        :param ids_time: list of the times that should be extracted
        :param ids_feature: list of the features that should be extracted
        :return: submask (as a torch tensor) extracted based on the indices
    """
    T, N_features = tensor.shape
    # If no identifiers have been specified, we use the whole data
    if ids_time is None:
        ids_time = [k for k in range(T)]
    if ids_feature is None:
        ids_feature = [k for k in range(N_features)]
    # Extract the relevant data in the mask
    subtensor = tensor.clone().detach()
    subtensor = subtensor[ids_time, :]
    subtensor = subtensor[:, ids_feature]
    return subtensor
