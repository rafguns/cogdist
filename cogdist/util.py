import pandas as pd

def fix_merged_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Fix CI overview files with merged cells

    Merged cells in Excel show up as NaN.

    """
    df = df.reset_index()
    df['level_0'] = df['level_0'].fillna(method='ffill')
    return df.set_index(['level_0', 'level_1'])
