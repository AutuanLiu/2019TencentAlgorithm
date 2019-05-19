import pandas as pd
import time, os, re
from functools import reduce
import numpy as np


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))


def isVaildDate(date_str):
    try:
        if ":" in date_str:
            time.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        else:
            time.strptime(date_str, "%Y-%m-%d")
        return True
    except:
        return False


def is_more(df, field):
    """判断是否存在多值"""
    ret, l = [], len(df)
    for i in range(l):
        if len(str(df.loc[i, field]).split(',')) > 1:
            ret.append(i)
    return ret


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    # fld_dtype = fld.dtype
    # if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
    #     fld_dtype = np.datetime64

    # if not np.issubdtype(fld_dtype, np.datetime64):
    #     df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    # targ_pre = re.sub('[Dd]ate$', '', fldname)
    targ_pre = 'crt_date'
    attr = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
        'Is_quarter_start', 'Is_year_end', 'Is_year_start'
    ]
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


def invalid_date(df, field):
    ret, l = [], len(df)
    df[field] = pd.to_datetime(df[field], unit='s')    # 转为日期
    df.reset_index(drop=True, inplace=True)    # 为了正常访问，重建索引
    for i in range(l):
        if not isVaildDate(str(df.loc[i, field])):
            ret.append(i)
    # 删除行
    new_df = df.drop(ret, axis=0)
    return new_df


def purge_pat_files(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def or_func(df_row):
    return reduce(lambda x, y: x | y, map(int, df_row.split(',')))


def split_vals(df_row, cols=None, field=None):
    features = df_row[field].split('|')
    if features[0] == 'all':
        df_row[cols] = -9    # 表示全部无限制
    elif len(features) == 1 and features[0] == '-999' or features[0] == '-999.0':
        df_row[cols] = -999    # 表示缺失
    elif len(features) >= 1:
        for fs in features:
            val = fs.split(':')
            tmp = 'device' if val[0] == 'os' else val[0]
            df_row[tmp] = val[1:][0]
    return df_row
