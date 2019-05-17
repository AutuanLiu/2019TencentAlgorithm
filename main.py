from functools import partial

import deepctr.models as ctr
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import plot_model

from models import AdaBound, AdamW, CyclicLR, huber_loss
from models.utils import *

# 工作路径设置
path = './data/'

# 数据读取
train_data = pd.read_csv(f'{path}train_set_final.csv', low_memory=False, encoding='utf-8')
test_data = pd.read_csv(f'{path}test_set_final.csv', low_memory=False, encoding='utf-8')

# 变量类型
sparse_features = [
    'ad_id', 'ad_type_id', 'ad_account_id', 'item_id', 'item_type', 'is_all_field', 'has_product_id', 'consuptionAbility', 'crt_dateYear', 'crt_dateMonth',
    'crt_dateWeek', 'crt_dateDay', 'crt_dateDayofweek', 'crt_dateDayofyear', 'crt_dateIs_month_end', 'crt_dateIs_month_start', 'crt_dateIs_quarter_end',
    'crt_dateIs_quarter_start', 'crt_dateIs_year_end', 'crt_dateIs_year_start', 'crt_dateHour', 'crt_dateElapsed'
]
dense_features = ['price']
multi_value_features = ['size', 'time', 'age', 'area', 'device', 'behavior', 'connectionType', 'gender', 'education', 'status', 'work']
multi_value_features_cnt = [5, 7, 1000, 2000, 8, 500, 7, 5, 10, 17, 8]

# 预测目标
target = train_data['target']

# settings
BATCH, EPOCH = 512, 1
cfg = {"hash_flag": True, "combiner": 'mean'}
padding_cfg = {"padding": 'post', "dtype": 'float32', "truncating": "post", "value": 0.}
model_name = 'xDeepFM'    # 'DeepFM', 'DIN'
model_settings = {"embedding_size": 8, "dnn_use_bn": True, "dnn_dropout": 0.0}

# 数值编码
# 稀疏特征
train_data = sparse_feature_encoding(train_data, sparse_features)
sparse_feat_list = sparse_feat_list_gen(train_data, sparse_features, mult=1, hash_flag=cfg["hash_flag"])

# 连续特征 标准化处理
train_data, scaler = dense_feature_scale(train_data, dense_features)
test_data, _ = dense_feature_scale(test_data, dense_features, scaler)
test_data.to_csv(f'{path}test_set_final1.csv', encoding='utf-8')
dense_feat_list = dense_feat_list_gen(train_data, dense_features, hash_flag=cfg["hash_flag"])

# 序列特征
padding_func = partial(pad_sequences, **padding_cfg)
sequence_feat_list, padding_feat_list = [], []
for feature, cnt in zip(multi_value_features, multi_value_features_cnt):
    squence, padding = single_multi_value_feature_encoding(train_data, feature, padding_func, max_feature_length=cnt, **cfg)
    sequence_feat_list.append(squence)
    padding_feat_list.append(padding)

# 模型输入
sparse_input = [train_data[feat.name].values for feat in sparse_feat_list]
dense_input = [train_data[feat.name].values for feat in dense_feat_list]
model_input = sparse_input + dense_input + padding_feat_list
feature_dim_dict = {"sparse": sparse_feat_list, "dense": dense_feat_list, "sequence": sequence_feat_list}

# 模型定义
model = getattr(ctr, model_name)(feature_dim_dict, task='regression', **model_settings)

# 模型结构与可视化
# print(model.summary())
plot_model(model, show_shapes=True, to_file=f'./imgs/{model_name}.png')

# 模型配置
adamw = AdamW(lr=3e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1, clipnorm=1.)
adabound = AdaBound(lr=3e-03, final_lr=0.1, gamma=1e-03, weight_decay=0.001, amsbound=False)
# clr = CyclicLR(scale_fn=lambda x: 1 / (5**(x * 0.0001)), scale_mode='iterations')
clr = CyclicLR(mode='triangular')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 模型拟合
model.compile(adabound, huber_loss, metrics=['mse'])
history = model.fit(model_input, target, batch_size=BATCH, epochs=EPOCH, verbose=1, validation_split=0.1, workers=4, callbacks=[early_stopping])

# 保存模型权重
model.save(f'./saved/{model_name}.h5')
model.save_weights(f'./saved/{model_name}_weights.h5')

# 可视化
# plt.plot(clr.history['iterations'], clr.history['lr'])
# plt.show()

# 损失函数
# plt.plot(history.epoch, history.history['loss'], 'g--', history.epoch, history.history['val_loss'], 'b--')
# plt.show()

# !TODO ！预测
multi_value_features_emb_sz = [10, 94, 76, 171, 5, 357, 5, 4, 6, 8, 5]

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

# 预测
preds = model.predict(model_input, batch_size=BATCH, verbose=1, workers=4)

# 提交文件生成
submission = pd.DataFrame(columns=['sample_id', 'preds'])
submission['sample_id'] = test_data['sample_id']
submission['preds'] = pd.Series(scale(preds))
submission.to_csv('../data/submission.csv', index=None, header=None, encoding='utf-8')

del model
