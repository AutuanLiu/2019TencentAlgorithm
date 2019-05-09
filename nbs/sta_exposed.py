import os, re
import modin.pandas as pd
from tqdm import trange


def save_csv(pdf):
    df = pd.DataFrame(columns=['曝光广告id', '曝光量'])
    df['曝光广告id'] = pd.Series(pdf.index.values, dtype='int64')
    df['曝光量'] = pd.Series(pdf['广告请求id'].values, dtype='float64')
    df.sort_values(by='曝光广告id', inplace=True)
    df.to_csv(f'../data/{date}_exposed.csv', index=None, encoding='utf-8')


def save_exposed_files(dir, pattern):
    fx = os.listdir(dir)
    for f, _ in zip(fx, trange(len(fx))):
        m = re.search(pattern, f)
        if m:
            data = pd.read_csv(f'../data/{f}', encoding='utf-8')
            data = data.groupby(by=['曝光广告id']).count()
            save_csv(data)

if __name__ == '__main__':
    pattern = r'^[^_]+_log.csv$'
    save_exposed_files('../data', pattern)
