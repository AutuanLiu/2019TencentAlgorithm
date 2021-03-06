import modin.pandas as pd
from fastai.tabular import *
from utils import isVaildDate, purge_pat_files
from tqdm import tqdm, trange

root = Path('../test_A')
u_data = root / 'user'
u_data.ls()
log_data = root / 'imps_log'
log_data.ls()

ad_static = pd.read_csv('../data/ad_static.csv', low_memory=False, encoding='utf-8')
c_sz = 102400
col_names1 = ['广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id', '广告请求时间_date']


def save_csv(row):
    date = str(row['广告请求时间_date'])
    pd.DataFrame(row).T.to_csv(f'../data/{date}_log.csv', mode='a', index=None, encoding='utf-8', header=False)


def invalid_date(df_row, field='广告请求时间'):
    """是否删除当前行,首先转为时间格式之后再行本操作"""
    if not isVaildDate(str(df_row[field])):
        df_row[field] = np.nan

    return df_row


purge_pat_files('../data', r'^[^_]+_log.csv$')

logs1 = pd.read_csv('../data/log_reduced.csv', encoding='utf-8', chunksize=c_sz)

for df, _ in zip(logs1, trange(1000)):
    df = pd.merge(df, ad_static, left_on='曝光广告id', right_on='广告id', how='inner')
    # 3. 去掉非法时间行
    df['广告请求时间'] = pd.to_datetime(df['广告请求时间'], unit='s')    # 转为日期
    df['广告请求时间_date'] = df['广告请求时间'].apply(lambda x: x.date())
    df = df[col_names1]

    # 1. 去空值
    df.dropna(axis=0, how='any', inplace=True)

    # 数据分割
    _ = df.apply(save_csv, axis=1)

print('done')
