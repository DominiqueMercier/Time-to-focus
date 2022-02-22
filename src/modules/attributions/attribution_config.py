import numpy as np

########################################
#### Default Attribution Config ########
########################################
config = {}

# working
config['FeatureAblation'] = {'execute': True}
config['FeaturePermutation'] = {'execute': True}
config['GradientShap'] = {'execute': True}
config['GuidedBackprop'] = {'execute': True}
config['InputXGradient'] = {'execute': True}
config['IntegratedGradients'] = {'execute': True}
config['KernelShap'] = {'execute': True}
config['Lime'] = {'execute': True}
config['Occlusion'] = {'execute': True}
config['Saliency'] = {'execute': True}
config['ShapleyValueSampling'] = {'execute': True}

config['Dynamask'] = {'execute': True}
config['Dynamask']['config'] = {
    'perturbation': 'GaussianBlur', 'area_list': np.linspace(0.05, 1.0, 10), 'n_epoch': 100}
