import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from deepctr.utils import VarLenFeat, SingleFeat
#定义稀疏编码函数
def sparse_feature_encoding(data,sparse_features_name):
    for feat in sparse_features_name:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data
#处理多值特征,该函数主要做分离

def split(x):
    print(x)
    key2index = {}
    pass
#     key_ans = x.split(',') #此处的','按具体任务设定
#     for key in key_ans:
#         if key not in key2index:
#             # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
#             key2index[key] = len(key2index) + 1
#     return list(map(lambda x: key2index[x], key_ans))

def single_multi_value_feature_encoding(data,multi_value_feature_name):
    print(multi_value_feature_name)
    feature_list = list(map(split, data[multi_value_feature_name].values))
    feature_length = np.array(list(map(len, feature_list)))
    max_len = max(feature_length)
    # Notice : padding=`post`
    feature_list = pad_sequences(feature_list, maxlen=max_len, padding='post', )
    return feature_list,max_len


def multi_value_feature_encoding(data,multi_value_feature_name):
    
    mulval_list=[]
    max_len_list=[]
    for feature in multi_value_feature_name:
        print(feature)
        feature_list,max_len=single_multi_value_feature_encoding(data,feature)
        mul_val_list.append(feature_list)
    return mul_val_list,max_len_list

def sequence_feature_acquire(max_len_list,multi_value_feature_name,hashing):
    key2index = {}
    output=[]
    if hashing:
        for max_len,feature in zip(max_len_list,multi_value_feature_name):
            sequence_feature = [VarLenFeat(feature, 100, max_len, 'mean', hash_flag=True,
                                   dtype="string")]  # Notice : value 0 is for padding for sequence input feature
            output.append(sequence_feature)
    else:
        for max_len,feature in zip(max_len_list,multi_value_feature_name):
            sequence_feature = [VarLenFeat(feature, len(
                                    key2index) + 1, max_len, 'mean')]  # Notice : value 0 is for padding for sequence input feature
            output.append(sequence_feature)
    return output

def sparse_feat_list_gen(data,sparse_features,hashing):
    if hashing:
        sparse_feat_list = [SingleFeat(feat, data[feat].nunique() * 5, hash_flag=True, dtype='string')
                    for feat in sparse_features]
    else:
        sparse_feat_list = [SingleFeat(feat, data[feat].nunique())
                        for feat in sparse_features]
    return sparse_feat_list