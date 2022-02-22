import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.metrics import infidelity
from modules.networks.model_utils import evaluate
from modules.utils.metrics import compute_compare, compute_continuity

sns.set()


class AttributionComparer:
    def __init__(self, save_memory=False, verbose=0):
        self.save_memory = save_memory
        self.verbose = verbose
        self.attributions = {}
        self.split_by = '_'
        self.use_last = True

    def get_attribution(self, name, group=None, remove_nan=True):
        if group is not None:
            true_name = name + self.split_by + \
                group if self.use_last else group + self.split_by + name
        else:
            true_name = name
        if type(self.attributions[true_name]) is str:
            return np.nan_to_num(np.load(self.approaches[true_name], allow_pickle=True)[1])
        else:
            return np.nan_to_num(self.attributions[true_name])

    def load_attributions(self, path, name=None):
        paths = [path] if type(path) is str else path
        if name is None:
            names = [os.path.split(p)[1].replace(
                '_attr.npy', '') for p in paths]
        elif type(name) is str:
            names = [name]
        else:
            names = name
        for (n, p) in zip(names, paths):
            attr = np.load(p, allow_pickle=True)[1]
            self.attributions[n] = attr if not self.save_memory else p
            if self.verbose:
                print('Loaded file:', p)

    def load_attribution_folder(self, folder):
        files = [f for f in os.listdir(folder) if '_attr.npy' in f]
        self.load_attributions(files)

    def group_by_method(self, split_by='_', use_last=True, sort=True):
        self.method_groups = {}
        self.split_by = split_by
        self.use_last = use_last
        for k in self.attributions:
            if use_last:
                identifier, method = ''.join(
                    k.split(split_by)[:-1]), k.split(split_by)[-1]
            else:
                method, identifier = k.split(
                    split_by)[0], ''.join(k.split(split_by)[1:])
            if method not in list(self.method_groups):
                self.method_groups[method] = []
            self.method_groups[method].append(identifier)
        for m in self.method_groups:
            self.method_groups[m] = sorted(self.method_groups[m])

    def compute_corr_to_first(self, map_ids=None, mode='spearmanr', histogram=10, perc=100):
        results = {}
        if map_ids is None:
            maps = np.arange(self.get_attribution(
                list(self.attributions)[0]).shape[0])
        elif type(map_ids) is int:
            maps = [map_ids]
        else:
            maps = map_ids
        for k in sorted(self.method_groups):
            corr = None
            for m in maps:
                attr_maps = np.array([self.get_attribution(an, group=k)[
                                     m] for an in self.method_groups[k]])
                if 'dtw' not in mode:
                    attr_maps = attr_maps.reshape(attr_maps.shape[0], -1)
                    if mode in ['spearmanr', 'pearsonr']:
                        keep = np.argsort(
                            attr_maps[0])[-int(attr_maps.shape[1] * perc / 100):]
                        attr_maps = attr_maps[:, keep]
                    if 'jaccard' in mode:
                        attr_maps = np.argsort(
                            attr_maps, -1)[:, -int(attr_maps.shape[1] * perc / 100):]
                if histogram > 0 and 'jaccard' not in mode:
                    attr_maps = np.array(
                        [np.histogram(d, histogram, range=(0, 1))[0] for d in attr_maps])
                entry = compute_compare(attr_maps[:1], attr_maps, mode=mode)
                entry = np.nan_to_num(entry)
                if corr is None:
                    corr = entry
                else:
                    corr += entry
            corr /= len(maps)
            results[k] = {}
            for sid, setup in enumerate(self.method_groups[k]):
                results[k][setup] = corr[0][sid]
        return results

    def compute_corr_mat(self, perc=100, key='B_', mode='spearmanr'):
        base_attr_names = [n for n in sorted(self.attributions) if key in n]
        base_attr_maps = [self.get_attribution(n) for n in base_attr_names]
        base_attr_corr_mat = compute_method_correlation(
            base_attr_maps, mode=mode, perc=perc)
        short_names = [n.replace(key, '') for n in base_attr_names]
        base_attr_corr_mat_pd = pd.DataFrame(
            data=base_attr_corr_mat, index=short_names, columns=short_names)
        return base_attr_corr_mat_pd

    def compute_method_continuity(self, key='B_'):
        base_attr_names = [n for n in sorted(self.attributions) if key in n]
        base_attr_maps = [self.get_attribution(n) for n in base_attr_names]
        continuties = {}
        for name, maps in zip(base_attr_names, base_attr_maps):
            continuties[name.replace(key, '')] = np.mean(
                compute_continuity(maps))
        return continuties

    def apply_attribution(self, data, names, idx=None, replace_strategy='zeros', perc=95, percentage=True, keep_smaller=True):
        modified_x = {}
        idx = idx if idx is not None else np.arange(data.shape[0])
        for n in names:
            attr = self.get_attribution(n)[idx]
            if not percentage:
                p = np.array([np.percentile(a, perc)
                             for a in attr]).reshape(-1, 1, 1)
                mask = (attr <= p).astype(
                    int) if keep_smaller else (attr >= p).astype(int)
            else:
                attr_c = attr.reshape(attr.shape[0], -1)
                sorted_attr = np.argsort(attr_c)
                n_keep = int(sorted_attr.shape[1] * perc / 100)
                if not keep_smaller:
                    sorted_attr = sorted_attr[:, ::-1]
                keep = sorted_attr[:, :n_keep]
                mask = np.zeros((sorted_attr.shape))
                for i, k in enumerate(keep):
                    mask[i, k] = 1
                mask = mask.reshape(attr.shape)
            if replace_strategy == 'zeros':
                modified_x[n] = data[idx] * mask
        return modified_x

    def compute_modified_accs(self, model, data, labels, names=None, replace_strategy='zeros', perc=95, percentage=True, keep_smaller=1, no_attr=True):
        modified_acc = {}
        if no_attr:
            cr = evaluate(model, data=data, labels=labels, return_dict=True)[
                'weighted avg']['f1-score']
            modified_acc['No_Attribution'] = float('%.4f' % (cr))
        for k in sorted(self.method_groups):
            modified_acc[k] = {}
            for m in self.method_groups[k]:
                attr_name = m + self.split_by + k
                if names is not None:
                    if attr_name not in names:
                        continue
                modified_x = self.apply_attribution(
                    data, [attr_name], replace_strategy=replace_strategy, perc=perc, percentage=percentage, keep_smaller=keep_smaller)
                cr = evaluate(model, data=modified_x[attr_name], labels=labels, return_dict=True)[
                    'weighted avg']['f1-score']
                modified_acc[k][m] = float('%.4f' % (cr))
            if len(list(modified_acc[k])) < 1:
                del modified_acc[k]
        return modified_acc

    def compute_modified_acc_dict(self, model, data, labels, names, percs, percentage=True):
        modified_accs_dict = {}
        for k, keep in zip([0, 1], ['larger', 'smaller']):
            modified_accs_dict[keep] = {}
            for p in percs[k]:
                accs = self.compute_modified_accs(
                    model, data, labels, names=names, replace_strategy='zeros', perc=p, percentage=percentage, keep_smaller=k, no_attr=False)
                if 'percs' not in list(modified_accs_dict[keep]):
                    modified_accs_dict[keep]['percs'] = [p]
                    modified_accs_dict[keep]['accs'] = {}
                    for m in sorted(accs):
                        modified_accs_dict[keep]['accs'][m] = [accs[m]['B']]
                else:
                    modified_accs_dict[keep]['percs'].append(p)
                    for m in sorted(accs):
                        modified_accs_dict[keep]['accs'][m].append(
                            accs[m]['B'])
        return modified_accs_dict

    def compute_infidelity(self, model, data, labels, scale=0.05, n_perturb_samples=10000):
        device = next(model.parameters()).device
        rng = np.random.RandomState(0)

        def perturb_fn(inputs):
            # noise = torch.tensor(np.random.normal(
            noise = torch.tensor(rng.normal(
                0, scale, inputs.shape)).float().to(device)
            return noise, inputs - noise

        data_tensor = torch.Tensor(data).to(device)
        t = [int(tar) for tar in labels]
        infidelities = {}
        for k in sorted(self.method_groups):
            infidelities[k] = {}
            for m in self.method_groups[k]:
                attr_name = m + self.split_by + k
                attr_tensor = torch.Tensor(
                    self.get_attribution(attr_name)).to(device)
                rng = np.random.RandomState(0)
                infid = infidelity(model, perturb_fn, data_tensor, attr_tensor,
                                   max_examples_per_batch=data.shape[0],
                                   n_perturb_samples=n_perturb_samples, normalize=True, target=t)
                infid = infid.detach().cpu().numpy()
                infidelities[k][m] = np.mean(infid)
        return infidelities

    def compute_agreement_dict(self, model, data, names, agree, verbose=0):
        device = next(model.parameters()).device
        outs = model(torch.Tensor(data).to(device))
        preds = torch.argmax(outs, dim=1).detach().cpu().numpy()
        ref_ag = [int(data.shape[0] * a / 100) for a in agree]

        agreement_dict = {'larger': {'agree': list(agree), 'ratios': {}}, 'smaller': {
            'agree': list(agree), 'ratios': {}}}
        for m in names:
            agreement_dict['larger']['ratios'][m.replace('B_', '')] = [
                None for _ in agree]
            agreement_dict['smaller']['ratios'][m.replace('B_', '')] = [
                None for _ in agree]
        for k, keep in zip([0, 1], ['larger', 'smaller']):
            rest_names = names.copy()
            for p in range(1, 101, 1):
                if len(rest_names) < 1:
                    break
                if verbose:
                    print('Keep: %s | Test percentage: %s | Methods left %s / %s' %
                          (keep, p, len(rest_names), len(names)))  # , end='\r')
                modified_x = self.apply_attribution(
                    data, rest_names, replace_strategy='zeros', perc=p, percentage=True, keep_smaller=k)
                for m_xname in sorted(modified_x):
                    m_outs = model(torch.Tensor(modified_x[m_xname]).to(device))
                    m_preds = torch.argmax(
                        m_outs, dim=1).detach().cpu().numpy()
                    curr_agree = np.sum(preds == m_preds)
                    m_real = m_xname.replace('B_', '')
                    for i in range(len(ref_ag)):
                        if curr_agree >= ref_ag[i] and agreement_dict[keep]['ratios'][m_real][i] is None:
                            agreement_dict[keep]['ratios'][m_real][i] = p
                    if agreement_dict[keep]['ratios'][m_real][-1] is not None and m_xname in rest_names:
                        rest_names.remove(m_xname)
        if verbose:
            print('All methods agree')
        return agreement_dict

    def plot_grid(self, sub_testX, index, methods=None, setups=None, not_show=False, save_path=None):
        methods = sorted(self.method_groups) if methods is None else methods
        setups = list(self.method_groups[methods[0]]
                      ) if setups is None else setups
        fig, ax = plt.subplots(figsize=(20, 10), nrows=len(
            methods)+1, ncols=len(setups)+1)
        ax[0][0].set_title('Original sample')
        ax[0][0].plot(sub_testX[index].T)
        ax[0][0].set_yticks([])
        ax[0][0].set_xticks([])
        for r, method in enumerate(methods):
            for c, setup in enumerate(setups):
                ax[r+1][c+1].plot(self.get_attribution(setup, method)[index].T)
                ax[r+1][c+1].set_yticks([])
                ax[r+1][c+1].set_xticks([])
                if r == 0:
                    ax[r+1][c+1].set_title(setup)
                    ax[r][c+1].set_visible(False)
            ax[r+1][1].set_ylabel(method, rotation='horizontal',
                                  ha='right', va='center')
            ax[r+1][0].set_visible(False)
        fig.tight_layout()

        if save_path is not None:
            fname = 'Attribution_comparison_grid.png'
            plt.savefig(os.path.join(save_path, fname), dpi=300,
                        bbox_inches='tight', pad_inches=0.1)

        if not not_show:
            plt.show()


def compute_method_correlation(attr_maps, perc=100, mode='spearmanr'):
    attr_corr_mat = None
    for i in range(attr_maps[0].shape[0]):
        maps = np.array([am[i].reshape(-1) for am in attr_maps])
        if 'jaccard' in mode:
            maps = np.argsort(maps, -1)[:, -int(maps.shape[1] * perc / 100):]
        entry = compute_compare(maps, None, mode=mode)
        entry = np.nan_to_num(entry)
        if attr_corr_mat is None:
            attr_corr_mat = entry
        else:
            attr_corr_mat += entry
    attr_corr_mat /= attr_maps[0].shape[0]
    return attr_corr_mat


def gather_all_attrs(dataset_name, exp_names=None, architectures=None, attr_paths=None, attr_names=None, exclude=None):
    files = glob('../../models/*/' + dataset_name +
                 '/**/*_attr.npy', recursive=True)
    if exp_names is not None:
        exp_names = exp_names.split(',') if type(
            exp_names) is str else exp_names
        files = [f for f in files if f.split('/')[-5] in exp_names]
    if architectures is not None:
        architectures = architectures.split(',') if type(
            architectures) is str else architectures
        files = [f for f in files if f.split('/')[-3] in architectures]
    if attr_paths is not None:
        attr_paths = attr_paths.split(',') if type(
            attr_paths) is str else attr_paths
        files = [f for f in files if f.split('/')[-2] in attr_paths]
    if attr_names is not None:
        attr_names = attr_names.split(',') if type(
            attr_names) is str else attr_names
        files = [f for f in files if f.split(
            '/')[-1].replace('_attr.npy', '') in attr_names]
    if exclude is not None:
        exclude = exclude.split(',') if type(exclude) is str else exclude
        files = [f for f in files if not np.max([e in f for e in exclude])]
    return np.array(sorted(files))


def shorten_attr_names(attr_paths):
    attr_names = []
    for path in attr_paths:
        splits = path.split('/')
        exp_name, architecture, method = splits[-5], splits[-3], splits[-1]
        name = 'B'
        if 'randomized_data' in exp_name:
            name = 'D'
        if '_randomize' in architecture:
            idx = int(architecture.split('-')[-1])
            name = 'R'
            if 'top' in architecture:
                name += 't{:02d}'.format(idx)
            if 'bottom' in architecture:
                name += 'b{:02d}'.format(idx)
            if 'ids' in architecture:
                name += 'i{:02d}'.format(idx)
        name += '_'
        name += method.replace('_attr.npy', '')
        attr_names.append(name)
    return np.array(attr_names)
