import os
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import modules.attributions.baselines.captum_approaches as cp
import numpy as np
import seaborn as sns
import torch
from captum.metrics import infidelity, sensitivity_max
from captum._utils.common import _format_input
from matplotlib import colors
from modules.attributions.dynamask.mask_group import MaskGroup
from modules.attributions.dynamask.perturbation import GaussianBlur

sns.set()


class ClassificationProcessor:
    def __init__(self, model, input_shape, attr_config={}, save_memory=False, attr_dir=None, load=True, verbose=0):
        self.device = next(model.parameters()).device
        self.model = model
        self.model.eval()
        self.input_shape = input_shape
        self.attr_config = deepcopy(attr_config)
        self.approaches = deepcopy(attr_config)
        self.save_memory = save_memory
        self.attr_dir = attr_dir
        self.load = load
        self.verbose = verbose
        # read existing attr dict
        if len(attr_config) > 0:
            if self.verbose:
                print('Prepared: ', end='')
            for name in sorted(self.approaches):
                self.init_approach(name, attr_dir=attr_dir)
                if self.verbose:
                    print('%s...' % name, end='')
            if self.verbose:
                print('Done')
        # load from folder
        if self.attr_dir is not None and self.load:
            names = [p.replace('_attr.npy', '')
                     for p in os.listdir(self.attr_dir) if '_attr.npy' in p]
            for name in names:
                self.load_attribution(name, self.attr_dir)

    def init_approach(self, name, attr_dir=None):
        if name == 'Deconvolution':
            self.approaches['Deconvolution']['method'] = cp.Deconvolution(
                self.model)
        elif name == 'DeepLift':
            self.approaches['DeepLift']['method'] = cp.DeepLift(
                self.model)
        elif name == 'DeepLiftShap':
            self.approaches['DeepLiftShap']['method'] = cp.DeepLiftShap(
                self.model)
        elif name == 'FeatureAblation':
            self.approaches['FeatureAblation']['method'] = cp.FeatureAblation(
                self.model)
        elif name == 'FeaturePermutation':
            self.approaches['FeaturePermutation']['method'] = cp.FeaturePermutation(
                self.model)
        elif name == 'GradientShap':
            self.approaches['GradientShap']['method'] = cp.GradientShap(
                self.model)
        elif name == 'GuidedBackprop':
            self.approaches['GuidedBackprop']['method'] = cp.GuidedBackprop(
                self.model)
        elif name == 'InputXGradient':
            self.approaches['InputXGradient']['method'] = cp.InputXGradient(
                self.model)
        elif name == 'IntegratedGradients':
            self.approaches['IntegratedGradients']['method'] = cp.IntegratedGradients(
                self.model)
        elif name == 'KernelShap':
            self.approaches['KernelShap']['method'] = cp.KernelShap(
                self.model)
        elif name == 'Lime':
            self.approaches['Lime']['method'] = cp.Lime(
                self.model)
        elif name == 'LRP':
            self.approaches['LRP']['method'] = cp.LRP(
                self.model)
        elif name == 'Occlusion':
            self.approaches['Occlusion']['method'] = cp.Occlusion(
                self.model)
        elif name == 'Saliency':
            self.approaches['Saliency']['method'] = cp.Saliency(self.model)
        elif name == 'ShapleyValueSampling':
            self.approaches['ShapleyValueSampling']['method'] = cp.ShapleyValueSampling(
                self.model)
        elif name == 'Dynamask':
            # create real perturbation based on key
            self.approaches['Dynamask']['fit_args'] = {
                'area_list': self.approaches['Dynamask']['config']['area_list'],
                'n_epoch': self.approaches['Dynamask']['config']['n_epoch']}
            # remove pert entry from static config
            del self.approaches['Dynamask']['config']['area_list']
            del self.approaches['Dynamask']['config']['n_epoch']
            self.approaches['Dynamask']['config']['perturbation'] = GaussianBlur(
                self.device)
            self.approaches['Dynamask']['method'] = MaskGroup(
                self.device, **self.approaches['Dynamask']['config'])
            self.approaches['Dynamask']['fit_args']['model'] = self.model

    def compute_infidelity(self, data, target, scale=0.05, n_perturb_samples=10000, verbose=0):
        rng = np.random.RandomState(0)

        def perturb_fn(inputs):
            #noise = torch.tensor(np.random.normal(
            noise = torch.tensor(rng.normal(
                0, scale, inputs.shape)).float().to(self.device)
            return noise, inputs - noise

        for name in sorted(list(self.approaches)):
            if not self.approaches[name]['execute']:
                continue
            data_tensor = torch.Tensor(data).to(self.device)
            if len(data.shape) < 3:
                data_tensor = data_tensor.unsqueeze(0)
                t = int(target)
            else:
                t = [int(tar) for tar in target]
            attr_tensor = torch.Tensor(
                self.approaches[name]['attr']).to(self.device)
            rng = np.random.RandomState(0)
            infid = infidelity(self.model, perturb_fn, data_tensor, attr_tensor,
                               max_examples_per_batch=data.shape[0],
                               n_perturb_samples=n_perturb_samples, normalize=True, target=t)
            infid = infid.detach().cpu().numpy()
            self.approaches[name]['infidelity'] = infid
            if verbose:
                print('Method: %s | Average infidelity: %.4f' %
                      (name, np.mean(infid)))

    def compute_sensitivity(self, data, target, perturb_radius=0.02, n_perturb_samples=10, verbose=0):
        rng = np.random.RandomState(0)

        def perturb_fn(inputs, perturb_radius=0.02):
            inputs = _format_input(inputs)
            perturbed_input = []
            for i in inputs:
                noise = torch.Tensor(rng.uniform(-perturb_radius, perturb_radius, i.size())).to(i.device)
                perturbed_input.append(i + noise)
            return tuple(perturbed_input)

        for name in sorted(list(self.approaches)):
            if not self.approaches[name]['execute']:
                continue
            data_tensor = torch.Tensor(data).to(self.device)
            if len(data.shape) < 3:
                data_tensor = data_tensor.unsqueeze(0)
                t = int(target)
            else:
                t = [int(tar) for tar in target]

            def forward_func(x, y, **kwargs):
                x = x[0]
                attr = []
                for i in range(x.size()[0]):
                    attr.append(self.approaches[name]['method'].attribute(
                        x[i].unsqueeze(0), y, **kwargs))
                attr = torch.cat(attr)
                return (attr,)

            sens_arr = []
            rng = np.random.RandomState(0)
            for i in range(data_tensor.size()[0]):
                if 'fit_args' not in list(self.approaches[name]):
                    sens = sensitivity_max(forward_func, data_tensor[i].unsqueeze(
                        0), perturb_func= perturb_fn,
                        perturb_radius=perturb_radius, n_perturb_samples=n_perturb_samples, y=[t[i]])
                else:
                    sens = sensitivity_max(forward_func, data_tensor[i].unsqueeze(
                        0), perturb_func= perturb_fn, perturb_radius=perturb_radius,
                        n_perturb_samples=n_perturb_samples, y=[t[i]], **self.approaches[name]['fit_args'])
                sens = sens.detach().cpu().numpy()
                sens = np.nan_to_num(sens)
                sens_arr.append(sens)
            self.approaches[name]['sensitivity'] = np.mean(sens_arr)
            if verbose:
                print('Method: %s | Average sensitivity: %.4f' %
                      (name, self.approaches[name]['sensitivity']))

    def process_attributions(self, data, target, folder=None):
        for name in sorted(list(self.approaches)):
            if not self.approaches[name]['execute']:
                continue
            data_tensor = torch.Tensor(data).to(self.device)
            if len(data.shape) < 3:
                data_tensor = data_tensor.unsqueeze(0)
                t = int(target)
            else:
                t = [int(tar) for tar in target]
            self.approaches[name]['attr'] = []
            start = time()
            for i in range(data_tensor.size()[0]):
                if self.verbose:
                    print('Method %s | Sample %s / %s' %
                          (name, i+1, data_tensor.size()[0]), end='\r')
                if 'fit_args' not in list(self.approaches[name]):
                    self.approaches[name]['attr'].append(self.approaches[name]['method'].attribute(
                        data_tensor[i].unsqueeze(0), [t[i]]))
                else:
                    self.approaches[name]['attr'].append(self.approaches[name]['method'].attribute(
                        data_tensor[i].unsqueeze(0), [t[i]], **self.approaches[name]['fit_args']))
            self.approaches[name]['time'] = time() - start
            self.approaches[name]['attr'] = torch.cat(
                self.approaches[name]['attr']).detach().cpu().numpy()
            if self.verbose:
                print('Approach: %s | Time: %.3fs' %
                      (name, self.approaches[name]['time']))
            if folder is not None:
                attr_path = self.save_attribution(
                    name, folder, return_path=True)
                if self.save_memory:
                    self.approaches[name]['attr'] = attr_path
            if self.verbose:
                print('Finished method', name)

    def get_attribution(self, name, remove_nan=True):
        if type(self.approaches[name]['attr']) is str:
            return np.nan_to_num(np.load(self.approaches[name]['attr'], allow_pickle=True)[1])
        else:
            return np.nan_to_num(self.approaches[name]['attr'])

    def save_attribution(self, name, path, return_path=False):
        np_file = os.path.join(path, name + '_attr.npy')
        data = np.array(
            [self.attr_config[name], self.approaches[name]['attr']], dtype=object)
        np.save(np_file, data)
        if self.verbose:
            print('Saved file:', np_file)
        if return_path:
            return np_file

    def load_attribution(self, name, path):
        np_file = os.path.join(path, name + '_attr.npy')
        config, attr = np.load(np_file, allow_pickle=True)
        self.attr_config[name] = config
        self.approaches[name] = deepcopy(self.attr_config[name])
        self.init_approach(name)
        if not self.save_memory:
            self.approaches[name]['attr'] = attr
        else:
            self.approaches[name]['attr'] = np_file
        if self.verbose:
            print('Loaded file:', np_file)

    def gather_dict(self, keys='time'):
        d = {}
        for name in sorted(self.approaches):
            if keys in list(self.approaches[name]) and type(keys) is str:
                d[name] = self.approaches[name][keys]
            else:
                for k in keys:
                    if k in list(self.approaches[name]):
                        if name not in list(d):
                            d[name] = {}
                        d[name][k] = self.approaches[name][k]
        return d

    def plot_approaches(self, data, index=0, not_show=False, save_path=None):
        approaches_exec = [a for a in sorted(
            self.approaches) if 'attr' in list(self.approaches[a])]
        rows = len(approaches_exec)
        cols = data.shape[1]
        if rows < 2:
            rows += 1
        if cols < 2:
            cols += 1
        fig, ax = plt.subplots(nrows=3*rows, ncols=cols,
                               figsize=(5*cols, 4*rows+1), sharex='row', sharey='row')
        fig.suptitle('Attribution methods for sample: ' + str(index))
        for c, name in enumerate(approaches_exec):
            vals = self.get_attribution(name)[index]
            if len(vals.shape) == 1:
                vals = [vals]
            for i in range(data.shape[1]):
                cmap = colors.LinearSegmentedColormap.from_list(
                    'incr_alpha', [(0, (*colors.to_rgb('C'+str(i)), 0)), (1, 'C'+str(i))])
                ax[c*3][0].set_title(name)
                ax[c*3][i].scatter(np.arange(data[index].shape[1]),
                                   data[index, i], c=vals[i], cmap=cmap)
                if c == 0:
                    ax[c*3][i].plot(data[index, i], c='C' +
                                    str(i), label='Channel ' + str(i))
                else:
                    ax[c*3][i].plot(data[index, i], c='C' + str(i))
                ax[c*3+1][i].hist(vals[i], np.arange(0, 1.05, 0.05), color='C'+str(i))
                ax[c*3+2][i].plot(vals[i], color='C'+str(i))
            if len(approaches_exec) < rows:
                for c in range(cols):
                    for i in range(3):
                        ax[len(approaches_exec) * 3 + i][c].set_visible(False)
            if data.shape[1] < cols:
                for r in range(3*rows):
                    ax[r][data.shape[1]].set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        if save_path is not None:
            fname = 'Attribution_Approaches_id-' + str(index) + '.png'
            plt.savefig(os.path.join(save_path, fname), dpi=300,
                        bbox_inches='tight', pad_inches=0.1)

        if not not_show:
            plt.show()
