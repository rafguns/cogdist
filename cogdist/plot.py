# coding: utf-8
import glob
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def errorbar_data(group_df, as_frame=False):
    group_df = group_df.sort_values('actual')
    data = group_df['actual']
    lohi = group_df[['lower', 'upper']]

    if as_frame:
        return data, abs(lohi.T - data)
    return data, abs(lohi.T - data).values


def plot_errorbars(ax, group_df, label):
    data, lohi = errorbar_data(group_df)

    n = len(data)
    x = np.arange(n)

    ax.set_title(label)
    ax.errorbar(x, data, yerr=lohi, fmt='k_')
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(list(data.index), rotation=45)

    return ax


def closest_pm(group_df, low_is_close):
    '''Get closest/most similar PM from `group_df`

    If `low_is_close` this is the one with the lowest value, otherwise the one
    with the highest value.

    '''
    pms = group_df.index.difference(['PanelTogether', 'Panel together'])
    if low_is_close:
        return group_df.reindex(index=pms).actual.idxmin()
    return group_df.reindex(index=pms).actual.idxmax()


def highlight_closest_pm(ax, group_df, low_is_close=True,
                         color='blue', alpha=0.1):
    pm = closest_pm(group_df, low_is_close)
    lower, upper = group_df.loc[pm][['lower', 'upper']]

    ax.axhspan(lower, upper, color=color, alpha=alpha)
    return ax


if __name__ == '__main__':
    data = {}
    for fname in glob.glob('*-ci-*.xlsx'):
        match = re.match(r'(.*)-ci-(.*).xlsx', fname)
        if match:
            data[match.groups()] = pd.read_excel(fname, index_col=[0, 1])
    print(data.keys())

    for k, df in data.items():
        # Merged cells show up as NaN. Fix this first
        df = df.reset_index()
        df['level_0'] = df['level_0'].fillna(method='ffill')
        df = df.set_index(['level_0', 'level_1'])

        data[k] = df

    fig = plt.figure()
    for (disc, method), df in data.items():
        low_is_close = method != 'wcs'
        for group_name in df.columns:
            group_df = df.unstack()[group_name]
            ax = fig.add_subplot(111)
            ax = plot_errorbars(ax, group_df, group_name)
            ax = highlight_closest_pm(ax, group_df, low_is_close)
            fig.savefig('{}-{}-{}-confidence-intervals.pdf'
                        .format(disc, group_name, method))
            fig.clf()
