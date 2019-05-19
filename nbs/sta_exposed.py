import os, re
import pandas as pd
from tqdm import trange
from utils import purge_pat_files


def save_csv(pdf, date):
    df = pd.DataFrame(columns=['曝光广告id', '曝光量'])
    df['曝光广告id'] = pd.Series(pdf.index.values, dtype='int64')
    df['曝光量'] = pd.Series(pdf['广告请求id'].values, dtype='float64')
    df.sort_values(by='曝光广告id', inplace=True)
    df.to_csv(f'../data/{date}_log_exposed.csv', index=None, encoding='utf-8')


def save_exposed_files(dir, pattern):
    fx = os.listdir(dir)
    for fs, _ in zip(fx, trange(len(fx))):
        m = re.search(pattern, fs)
        if m:
            date = m.group(1)
            pdf = pd.read_csv(f'../data/{fs}', encoding='utf-8')
            pdf = pdf.groupby(by=['曝光广告id']).count()
            save_csv(pdf, date)


if __name__ == '__main__':
    pattern = r'^([^_]+)_log_exposed.csv$'
    pattern1 = r'^([^_]+)_log.csv$'
    purge_pat_files('../data', pattern)    # 先删除曾经创建的同名文件
    save_exposed_files('../data', pattern1)
    print('done!')
