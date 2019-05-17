from functools import partial

import deepctr.models as ctr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import load_model

from models.utils import *

# 工作路径设置
path = './data/'

# 数据读取
test_data = pd.read_csv(f'{path}test_set_final1.csv', low_memory=False, encoding='utf-8')

# 变量类型
sparse_features = [
    'ad_id', 'ad_type_id', 'ad_account_id', 'item_id', 'item_type', 'is_all_field', 'has_product_id', 'consuptionAbility', 'crt_dateYear', 'crt_dateMonth',
    'crt_dateWeek', 'crt_dateDay', 'crt_dateDayofweek', 'crt_dateDayofyear', 'crt_dateIs_month_end', 'crt_dateIs_month_start', 'crt_dateIs_quarter_end',
    'crt_dateIs_quarter_start', 'crt_dateIs_year_end', 'crt_dateIs_year_start', 'crt_dateHour', 'crt_dateElapsed'
]
dense_features = ['price']
multi_value_features = ['size', 'time', 'age', 'area', 'device', 'behavior', 'connectionType', 'gender', 'education', 'status', 'work']
multi_value_features_cnt = [5, 7, 1000, 2000, 8, 500, 7, 5, 10, 17, 8]
multi_value_features_emb_sz = [10, 94, 76, 171, 5, 357, 5, 4, 6, 8, 5]

# settings
cfg = {"hash_flag": False, "combiner": 'mean'}
padding_cfg = {"padding": 'post', "dtype": 'float32', "truncating": "post", "value": 0.}
model_name = 'xDeepFM'    # 'DeepFM', 'DIN'

# 数值编码
# 稀疏特征
test_data = sparse_feature_encoding(test_data, sparse_features)
sparse_feat_list = sparse_feat_list_gen(test_data, sparse_features, mult=1, hash_flag=cfg["hash_flag"])

# 连续特征
dense_feat_list = dense_feat_list_gen(test_data, dense_features, hash_flag=cfg["hash_flag"])

# 序列特征
padding_func = partial(pad_sequences, **padding_cfg)
padding_feat_list = []
for feature, emb_sz, cnt in zip(multi_value_features, multi_value_features_emb_sz, multi_value_features_cnt):
    _, padding = single_multi_value_feature_encoding(test_data, feature, padding_func, sequence_dim=emb_sz, max_feature_length=cnt, **cfg)
    padding_feat_list.append(padding)

# 模型输入
sparse_input = [test_data[feat.name].values for feat in sparse_feat_list]
dense_input = [test_data[feat.name].values for feat in dense_feat_list]
model_input = sparse_input + dense_input + padding_feat_list

# 模型加载
model = load_model(f'./saved/{model_name}.h5')

# 预测
preds = model.predict(model_input, batch_size=1, verbose=1, workers=4)

# 提交文件生成
submission = pd.DataFrame(columns=['sample_id', 'preds'])
submission['sample_id'] = test_data['sample_id']
submission['preds'] = pd.Series(scale(preds))
submission.to_csv('../data/submission.csv', index=None, header=None, encoding='utf-8')
