from functools import partial

import numpy as np
import pandas as pd
from deepctr.models import DeepFM, xDeepFM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from models import AdaBound, AdamW, CyclicLR
from models.utils import *

# 工作路径设置
path = './data/'

# 数据读取
test_data = pd.read_csv(f'{path}test_set_final.csv', low_memory=False, encoding='utf-8')

# 变量类型
sparse_features = [
    'ad_id', 'ad_type_id', 'ad_account_id', 'item_id', 'item_type', 'is_all_field', 'has_product_id', 'consuptionAbility', 'crt_dateYear', 'crt_dateMonth',
    'crt_dateWeek', 'crt_dateDay', 'crt_dateDayofweek', 'crt_dateDayofyear', 'crt_dateIs_month_end', 'crt_dateIs_month_start', 'crt_dateIs_quarter_end',
    'crt_dateIs_quarter_start', 'crt_dateIs_year_end', 'crt_dateIs_year_start', 'crt_dateHour', 'crt_dateElapsed'
]
dense_features = ['price']
multi_value_features = ['size', 'time', 'age', 'area', 'device', 'behavior', 'connectionType', 'gender', 'education', 'status', 'work']
multi_value_features_cnt = [4, 7, 10, 10, 8, 10, 7, 5, 10, 17, 8]

# settings
cfg = {"hashing": False, "p_combiner": 'mean', "padding": 'post', "p_dtype": 'float64', "p_truncating": "post", "p_value": 0.}

# 数值编码
# 稀疏特征
test_data = sparse_feature_encoding(test_data, sparse_features)
sparse_feat_list = sparse_feat_list_gen(test_data, sparse_features, mult=1, hashing=cfg["hashing"])

# 连续特征
dense_feat_list = dense_feat_list_gen(test_data, dense_features, hashing=cfg["hashing"])

# 序列特征
padding_func = partial(pad_sequences, dtype=cfg["p_dtype"], padding=cfg["padding"], truncating=cfg["p_truncating"], value=cfg["p_value"])
keys, sequence_feat_list, padding_feat_list = {}, [], []
for feature, cnt in zip(multi_value_features, multi_value_features_cnt):
    squence, padding = single_multi_value_feature_encoding(test_data,
                                                           feature,
                                                           padding_func,
                                                           keys,
                                                           sequence_dim=None,
                                                           max_feature_length=cnt,
                                                           combiner=cfg["p_combiner"],
                                                           hashing=cfg["hashing"])
    sequence_feat_list.append(squence)
    padding_feat_list.append(padding)

# 模型输入
sparse_input = [test_data[feat.name].values for feat in sparse_feat_list]
dense_input = [test_data[feat.name].values for feat in dense_feat_list]
sequence_input = padding_feat_list
model_input = sparse_input + dense_input + sequence_input

# 模型加载
model = load_model('./models/xdeepfm.h5')
print(model.summary())

# 预测
preds = model.predict(model_input, batch_size=1, verbose=1, workers=4)

# 提交文件生成
submission = pd.DataFrame(columns=['sample_id', 'preds'])
submission['sample_id'] = test['sample_id']
submission['preds'] = pd.Series(preds).apply(partial(round, ndigits=4))
submission.to_csv('../data/submission.csv', index=None, header=None, encoding='utf-8')
