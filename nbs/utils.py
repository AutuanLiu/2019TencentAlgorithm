import modin as pd


def join_df(left, right, left_on, how='inner', right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on, suffixes=("", suffix))
