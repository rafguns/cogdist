import sys

import pandas as pd

from .util import fix_merged_cells


def intervals_overlap(a, b):
    a_lower, a_upper = a
    b_lower, b_upper = b

    return not (a_upper <= b_lower or b_upper <= a_lower)


if __name__ == '__main__':
    df = pd.read_excel(sys.argv[1], index_col=[0, 1])
    df = fix_merged_cells(df)
    # We're only interested in individual PMs
    df = df.drop(['Panel Together', 'PanelTogether', 'Panel together'],
                 errors='ignore')

    asc = not (len(sys.argv) == 3 and sys.argv[2] == 'top')

    for group in df.columns:
        print(group)
        group_df = df.unstack()[group].sort_values('actual', ascending=asc)
        bottom_interval = group_df.iloc[0][['lower', 'upper']]
        print('-', group_df.index[0])

        for idx, row in group_df.iloc[1:].iterrows():
            interval = row[['lower', 'upper']]
            if intervals_overlap(bottom_interval, interval):
                print('-', idx)
