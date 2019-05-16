from functools import partial

import numpy as np
import pandas as pd
from deepctr.models import DeepFM, xDeepFM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from models import AdaBound, AdamW, CyclicLR
from models.utils import *

# 工作路径设置
path = './data/'

# 数据读取
train_data = pd.read_csv(f'{path}train_set_final.csv', low_memory=False, encoding='utf-8')

# 变量类型
sparse_features = [
    'ad_id', 'ad_type_id', 'ad_account_id', 'item_id', 'item_type', 'is_all_field', 'has_product_id', 'consuptionAbility', 'crt_dateYear', 'crt_dateMonth',
    'crt_dateWeek', 'crt_dateDay', 'crt_dateDayofweek', 'crt_dateDayofyear', 'crt_dateIs_month_end', 'crt_dateIs_month_start', 'crt_dateIs_quarter_end',
    'crt_dateIs_quarter_start', 'crt_dateIs_year_end', 'crt_dateIs_year_start', 'crt_dateHour', 'crt_dateElapsed'
]
dense_features = ['price']
multi_value_features = ['size', 'time', 'age', 'area', 'device', 'behavior', 'connectionType', 'gender', 'education', 'status', 'work']
multi_value_features_cnt = [4, 7, 10, 10, 8, 10, 7, 5, 10, 17, 8]

# 预测目标
target = train_data['target']

# settings
cfg = {"hashing": False, "p_combiner": 'mean', "padding": 'post', "p_dtype": 'float64', "p_truncating": "post", "p_value": 0.}

# 数值编码
# 稀疏特征
train_data = sparse_feature_encoding(train_data, sparse_features)
sparse_feat_list = sparse_feat_list_gen(train_data, sparse_features, mult=1, hashing=cfg["hashing"])

# 连续特征
dense_feat_list = dense_feat_list_gen(train_data, dense_features, hashing=cfg["hashing"])

# 序列特征
padding_func = partial(pad_sequences, dtype=cfg["p_dtype"], padding=cfg["padding"], truncating=cfg["p_truncating"], value=cfg["p_value"])
keys, sequence_feat_list, padding_feat_list = {}, [], []
for feature, cnt in zip(multi_value_features, multi_value_features_cnt):
    squence, padding = single_multi_value_feature_encoding(train_data,
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
sparse_input = [train_data[feat.name].values for feat in sparse_feat_list]
dense_input = [train_data[feat.name].values for feat in dense_feat_list]
sequence_input = padding_feat_list
model_input = sparse_input + dense_input + sequence_input

# 模型定义
model = xDeepFM({
    "sparse": sparse_feat_list,
    "dense": dense_feat_list,
    "sequence": sequence_feat_list
},
                embedding_size=10,
                dnn_use_bn=True,
                dnn_dropout=0.1,
                task='regression')

print(model.summary())

# 模型配置
adamw = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)
# adabound = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=0., amsbound=False)
clr = CyclicLR(scale_fn=lambda x: 1 / (5**(x * 0.0001)), scale_mode='iterations')
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

# 模型拟合
model.compile(adamw, "mse", metrics=['mse'])
history = model.fit(model_input, target, batch_size=32, epochs=500, verbose=1, validation_split=0.2, workers=4, callbacks=[clr, early_stopping])

# 保存模型权重
model.save('./models/xdeepfm.h5')
model.save_weights('./models/xdeepfm_weights.h5')
