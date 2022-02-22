import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def plot_correlations(correlations, mode='spearmanr', perc=100, not_show=False, save_path=None):
    methods = sorted(correlations)

    x = np.arange(len(methods))  # the label locations
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 4))
    vals = []
    for mid, setup in enumerate(list(correlations[methods[0]])[1:]):
        vals.append([correlations[m][setup] for m in methods])
        off = (-width * len(vals[-1])) / 2 + mid * width
        ax.bar(x + off, vals[-1], width, label=setup)

    ax.set_ylabel('Correlation ' + mode)
    ax.set_title('Correlation w.r.t. original attribution')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(vals))
    fig.tight_layout()

    if save_path is not None:
        fname = 'Correlation_' + mode + '_perc-' + str(perc) + '.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_modified_accs(modified_accs, perc=95, keep_smaller=True, not_show=False, save_path=None):
    methods = sorted(list(modified_accs)[1:])

    x = np.arange(len(methods))  # the label locations
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 4))
    vals = []
    for mid, setup in enumerate(modified_accs[methods[0]]):
        vals.append([modified_accs[m][setup] for m in methods])
        off = (-width * len(vals[-1])) / 2 + mid * width
        ax.bar(x + off, vals[-1], width, label=setup)

    ax.axhline(y=modified_accs['No_Attribution'],
               linestyle='--', label='No Attribution')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy drop using masked samples')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.08), ncol=int(np.ceil(len(vals)/2)))
    fig.tight_layout()

    if save_path is not None:
        fname = 'Modified_Accuracies_perc-' + \
            str(perc) + '_keep_smaller-' + str(keep_smaller) + '.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_infidalities(infidalities, not_show=False, save_path=None):
    methods = sorted(infidalities)

    x = np.arange(len(methods))  # the label locations
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 4))
    vals = []
    for mid, setup in enumerate(list(infidalities[methods[0]])):
        vals.append([infidalities[m][setup] for m in methods])
        off = (-width * len(vals[-1])) / 2 + mid * width
        ax.bar(x + off, vals[-1], width, label=setup)

    ax.set_ylabel('Infidelity')
    ax.set_title('Infidelity of the methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(np.min(np.array(vals).reshape(-1))*0.99)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(vals))
    fig.tight_layout()

    if save_path is not None:
        fname = 'Infidelity.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_correlation_matrix(corr_pd, mode='spearmanr', perc=100, not_show=False, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Correlation Matrix: ' + mode)
    mask = 1 - np.triu(np.ones_like(corr_pd, dtype=bool))
    sns.heatmap(corr_pd, mask=mask, xticklabels=corr_pd.columns,
                yticklabels=corr_pd.columns, annot=True, fmt='.2f', cmap='Blues', square=True)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()

    if save_path is not None:
        fname = 'Correlation_Matrix_' + mode + \
            '_perc-' + str(perc) + '_key-B.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_masked_dict(masked_dict, percentage=True, not_show=False, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 4), ncols=2)

    for m in sorted(masked_dict['larger']['accs']):
        ax[0].plot(masked_dict['larger']['percs'], masked_dict['larger']
                   ['accs'][m], linestyle='-', marker='x', label=m)
        ax[1].plot(masked_dict['smaller']['percs'], masked_dict['smaller']
                   ['accs'][m], linestyle=':', marker='x')

        ax[0].set_xlabel(
            'Percentile data' if not percentage else 'Percentage data')
        ax[1].set_xlabel(
            'Percentile data' if not percentage else 'Percentage data')
        ax[0].set_ylabel('Accuracy')
        ax[1].set_ylabel('Accuracy')
        ax[0].set_title('Accuracy keep larger importance')
        ax[1].set_title('Accuracy keep small importance')
    fig.legend(loc='lower center', bbox_to_anchor=(.5, -0.15), ncol=7)
    fig.tight_layout()

    if save_path is not None:
        fname = 'Modified_Accuracies_Matrix.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_bars(bar_dict, key, ylabel='Value', level=1, not_show=False, save_path=None):
    methods = sorted(bar_dict)
    vals = np.array([bar_dict[m] if level == 1 else bar_dict[m][key]
                    for m in methods])

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_title('Comparison: ' + key)
    ax.bar(methods, vals)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save_path is not None:
        fname = 'Attribution_' + key + '.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()


def plot_agreements(agreements, not_show=False, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 4))
    methods = sorted(agreements['larger']['ratios'])
    koff = len(agreements['larger']['agree'])
    x = np.arange(len(methods))  # the label locations
    width = 0.05  # the width of the bars
    vals, barId = [], 0
    for kid, keep in enumerate(agreements):
        for mid, setup in enumerate(agreements[keep]['agree']):
            vals.append([agreements[keep]['ratios'][m][mid] for m in methods])
            off = -width * koff + barId * width + width/2
            ax.bar(x + off, vals[-1], width, label='%s%% %sest attr.' %
                   (int(setup), 'high' if keep == 'larger' else 'low'))
            barId += 1

    ax.set_ylabel('Percentage data')
    ax.set_title('Agreement adding data points')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(np.min(vals) * 0.99)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=len(vals))
    fig.tight_layout()

    if save_path is not None:
        fname = 'Modified_Agreement_Matrix.png'
        plt.savefig(os.path.join(save_path, fname), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)

    if not not_show:
        plt.show()
