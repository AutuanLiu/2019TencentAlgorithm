import modin.pandas as pd

col = ['广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id']
col_names = [
    '广告请求id', '广告请求时间', '广告位id', '用户id', '曝光广告id', '曝光广告素材尺寸', '曝光广告出价bid', '曝光广告pctr', '曝光广告quality_ecpm',
    '曝光广告totalEcpm'
]
c_sz = 102400

print('start!')
logs = pd.read_csv('../test_A/imps_log/totalExposureLog.out', sep='\t', header=None, names=col_names)
logs = logs[col]
logs.to_csv('../data/log_reduced.csv', index=None, encoding='utf-8', chunksize=c_sz)
print('fin!')
