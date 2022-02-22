import captum.attr as cp
import torch


def normalize(tensor):
    n_samples = tensor.size()[0]
    mi, _ = torch.min(tensor.view(n_samples, -1), dim=1)
    tensor -= mi.view(n_samples, 1, 1)
    ma, _ = torch.max(tensor.view(n_samples, -1), dim=1)
    tensor /= ma.view(n_samples, 1, 1)
    return tensor


class FeatureAblation:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.FeatureAblation(self.forward_func)
        attr = exp.attribute(x, target=y)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class FeaturePermutation:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.FeaturePermutation(self.forward_func)
        if x.size()[0] > 1:
            attr = exp.attribute(x, target=y)
        else:
            x_modified = torch.cat([x, torch.zeros(x.size()).to(x.device)])
            attr = exp.attribute(x_modified, target=y*2)[:1]
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class GradientShap:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.GradientShap(
            forward_func=self.forward_func)
        # baselines w.r.t. standardized data
        base = torch.cat([-1*x, 0*x, 1*x])
        attr = exp.attribute(x, target=y, baselines=base)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class GuidedBackprop:
    def __init__(self, model):
        self.model = model

    def attribute(self, x, y, norm=True):
        exp = cp.GuidedBackprop(self.model)
        attr = exp.attribute(x, target=y)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class InputXGradient:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.InputXGradient(forward_func=self.forward_func)
        attr = exp.attribute(x, target=y).contiguous()
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class IntegratedGradients:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.IntegratedGradients(forward_func=self.forward_func)
        base = torch.mean(x, dim=-1, keepdim=True) * \
            torch.ones(x.size()).to(x.device)  # base of the input
        attr = exp.attribute(x, target=y, baselines=base)
        if norm:
            # absolute corresponds to importance
            return normalize(torch.abs(attr))
        return attr


class KernelShap:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.KernelShap(self.forward_func)
        # n_samples is set according to the KernelShap paper authors
        n_samples = 2 * x.size()[1] * x.size()[2] + 2048
        attr = exp.attribute(x, target=y, n_samples=n_samples).contiguous()
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class Lime:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.Lime(self.forward_func)
        attr = exp.attribute(x, target=y, n_samples=1000).contiguous()
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr
    

class Occlusion:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.Occlusion(forward_func=self.forward_func)
        base = torch.mean(x, dim=-1, keepdim=True) * \
            torch.ones(x.size()).to(x.device)  # base of the input
        # set windows to 3 as this is the smallest possible pattern
        attr = exp.attribute(
            x, target=y, sliding_window_shapes=(1, 3), baselines=base)
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr


class Saliency:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.Saliency(forward_func=self.forward_func)
        # absolute values for importance
        attr = exp.attribute(x, target=y, abs=True)
        if norm:
            return normalize(attr)
        return attr


class ShapleyValueSampling:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    def attribute(self, x, y, norm=True):
        exp = cp.ShapleyValueSampling(self.forward_func)
        base = torch.mean(x, dim=-1, keepdim=True) * \
            torch.ones(x.size()).to(x.device)  # base of the input
        # set n_samples based on accuracy and time trade-off
        attr = exp.attribute(x, target=y, baselines=base, n_samples=25).contiguous()
        if norm:
            return normalize(torch.abs(attr))  # absolute for importance
        return attr
