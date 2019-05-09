import modin.pandas as pd
from fastai.tabular import *
from utils import isVaildDate, is_more, purge_pat_files
from tqdm import tqdm, trange

root = Path('../test_A')
u_data = root/'user'
u_data.ls()
log_data = root/'imps_log'
log_data.ls()

ad_static = pd.read_csv('../data/ad_static.csv', low_memory=False, encoding='utf-8')

col_names = ['广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id', '曝光广告素材尺寸', '曝光广告出价bid',
             '曝光广告pctr', '曝光广告quality_ecpm', '曝光广告totalEcpm']


c_sz = 102400

def save_csv(row):
    date = str(row['广告请求时间_date'])  
    pd.DataFrame(row).T.to_csv(f'../data/{date}_log.csv', mode='a', index=None, encoding='utf-8', header=False)

def invalid_date(df_row, field='广告请求时间'):
    """是否删除当前行,首先转为时间格式之后再行本操作"""
    if not isVaildDate(str(df_row[field])):
        df_row[field] = np.nan
    
    return df_row


logs1 = pd.read_csv(log_data/'totalExposureLog.out', sep='\t', header=None, names=col_names, chunksize=c_sz)

purge_pat_files('../data', r'^[^_]+_log.csv$')

for df, _ in zip(logs1, trange(1000)):
    df = pd.merge(df, ad_static, left_on='曝光广告id', right_on='广告id', how='inner')
    # 3. 去掉非法时间行
    df['广告请求时间'] = pd.to_datetime(df['广告请求时间'], unit='s')  # 转为日期
    df = df.apply(invalid_date, axis=1)
    
    # 1. 去空值
    df.dropna(axis=0, how='any', inplace=True)
    # 2. 去重 所有列相同
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    # 4. 数据类型转换
    # 暂无
    # 数据分割
    df['广告请求时间_date'] = df['广告请求时间'].apply(lambda x: x.date())
    _ = df.apply(save_csv, axis=1)
#     df.to_csv('../data/ad_logs.csv', mode='a', index=None, encoding='utf-8', header=False)

col_names1 = ['广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id', '曝光广告素材尺寸', '曝光广告出价bid',
       '曝光广告pctr', '曝光广告quality_ecpm', '曝光广告totalEcpm', '广告id', '创建时间',
       '广告账户id', '商品id', '商品类型', '广告行业id', '素材尺寸', '广告请求时间_date']

def read_files(dir, pattern):
    fx = os.listdir(dir)
    for f, _ in zip(fx, trange(len(fx))):
        if re.search(pattern, f):
            data = pd.read_csv(f'../data/{f}', sep='\t', header=None, names=col_names1)
            data.drop_duplicates(subset=None, keep='first', inplace=True)
            data.to_csv(f'../data/{f}', mode='w', index=None, encoding='utf-8')
            
read_files('../data', r'^[^_]+_log.csv$')