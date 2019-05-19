import os, re
import pandas as pd
from tqdm import tqdm, trange


def read_files(dir, pattern, cols=None):
    fx = os.listdir(dir)
    for fs, _ in zip(fx, trange(len(fx))):
        if re.search(pattern, fs):
            data = pd.read_csv(f'../data/{fs}', header=None, names=cols, encoding='utf-8')    # 前处理
            # data = pd.read_csv(f'../data/{fs}', encoding='utf-8')  # 后处理
            data.drop_duplicates(subset=None, keep='first', inplace=True)
            data.to_csv(f'../data/{fs}', index=None, encoding='utf-8')


if __name__ == "__main__":
    pattern = r'^[^_]+_log.csv$'
    cols = ['广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id', '广告请求时间_date']
    read_files('../data', pattern, cols)
    print('done!')
