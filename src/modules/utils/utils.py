import json
import os
import pickle

from sklearn.metrics import classification_report


def compute_classification_report(gt, preds, save=False, verbose=0, store_dict=False):
    s = classification_report(gt, preds, digits=4)
    if verbose:
        print(s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)
        if verbose:
            print('Save Location:', save)
        if store_dict:
            cr_dict = classification_report(
                gt, preds, digits=4, output_dict=True)
            with open(save.replace('.txt', '.pickle'), 'wb') as f:
                pickle.dump(cr_dict, f)


def get_files(folder, contains=None):
    paths = []
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            if contains is not None and contains not in file:
                continue
            paths.append(full_path)
    return paths


def maybe_create_dirs(dataset_name, root='../../', dirs=['models', 'results'], exp=None, return_paths=False, verbose=0):
    paths = []
    for d in dirs:
        if exp is None:
            tmp = os.path.join(root, d, dataset_name)
        else:
            tmp = os.path.join(root, d, exp, dataset_name)
        paths.append(tmp)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
            if verbose:
                print('Created directory:', tmp)
        elif verbose:
            print('Found existing directory:', tmp)
    if return_paths:
        return paths


def get_pretty_dict(dictionary, sort=False, save=None, verbose=0):
    s =json.dumps(str(dictionary), sort_keys=sort, indent=4)
    if verbose:
        print(s)
    if save is not None:
        with open(save, "w") as f:
            f.write(s)
