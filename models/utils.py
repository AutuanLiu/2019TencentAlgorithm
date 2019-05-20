import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from deepctr.utils import SingleFeat, VarLenFeat

# 可以使用的模型集合以及发行年份
Models = {
    "FGCNN": 201904,
    "AutoInt": 201810,
    "DIEN": 201809,
    "xDeepFM": 201803,
    "NFM": 201708,
    "AFM": 201708,
    "DCN": 201708,
    "DIN": 201706,
    "DeepFM": 201703,
    "PNN": 201611,
    "FNN": 201601,
    "CCPM": 201510,
}


def relu(x: np.ndarray) -> np.ndarray:
    out = x if isinstance(x, np.ndarray) else np.array(x)
    return np.clip(out, 0, None)


def scale(x: np.ndarray) -> np.ndarray:
    out = relu(x)
    out = np.exp(out) - 1
    out = np.round(out, decimals=4)
    return out


def emb_sz_rule1(n_cat: int) -> int:
    return min(50, (n_cat // 2) + 1)


def emb_sz_rule(n_cat: int) -> int:
    return min(600, round(1.6 * n_cat**0.56))


def sparse_feature_encoding(data, sparse_features_names):
    """稀疏编码函数"""

    for feat in sparse_features_names:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data


def dense_feature_scale(data, dense_features_names, scaler=None):
    """连续变量放缩"""

    scaler = scaler if scaler else StandardScaler().fit(data[dense_features_names])
    data[dense_features_names] = scaler.transform(data[dense_features_names])
    return data, scaler


def single_multi_value_feature_encoding(data, feature, padding_func, seq_dim=None, max_len=None, **kwargs):
    """单个特征操作
    
    Arguments:
        data (pd.DataFrame): -- 原始数据
        feature (str): -- 多值特征名
        padding_func (function) -- padding 函数
        max_len (int) -- 自定义特征长度
        seq_dim (int) -- 序列维度
    
    Returns:
        sequence_feature  -- padding 后的特征
    """

    key2index = {}

    def split(x, sep=','):
        """处理多值特征,该函数主要做分离"""

        x = str(x) if not isinstance(x, str) else x
        key_ans = x.split(sep)
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    feature_list = list(map(split, data[feature].values))    # 分割
    max_length = max_len if max_len else max(list(map(len, feature_list)))
    # padding 对齐
    padding_feature = padding_func(feature_list, max_length)
    dim = seq_dim if seq_dim else emb_sz_rule(len(key2index) + 1)
    print(f'len_unique： {len(key2index) + 1: 8d}, emb_sz： {dim: 8d}, max_len： {max_length: 8d}')
    del key2index
    sequence_feature = VarLenFeat(feature, dim, max_length, dtype="float32", **kwargs)
    return sequence_feature, padding_feature


def sparse_feat_list_gen(data, sparse_features, emb_rule=True, hash_flag=False):
    dim = {feat: emb_sz_rule(data[feat].nunique()) if emb_rule else data[feat].nunique() for feat in sparse_features}
    return [SingleFeat(feat, dim[feat], hash_flag=hash_flag, dtype='float32') for feat in sparse_features]


def dense_feat_list_gen(data, dense_features, hash_flag=False):
    return [SingleFeat(feat, 0, hash_flag=hash_flag, dtype='float32') for feat in dense_features]
