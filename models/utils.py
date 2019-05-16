import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepctr.utils import SingleFeat, VarLenFeat


def sparse_feature_encoding(data, sparse_features_names):
    """稀疏编码函数"""

    for feat in sparse_features_names:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    return data


def single_multi_value_feature_encoding(data, feature, padding_func, key2index={}, sequence_dim=None, max_feature_length=None, combiner='mean', hashing=False):
    """单个特征操作
    
    Arguments:
        data (pd.DataFrame): -- 原始数据
        feature (str): -- 多值特征名
        padding_func (function) -- padding 函数
        max_feature_length (int) -- 自定义长度
        sequence_dim (int) -- 序列特征维度
        combiner (str) -- defaults to 'mean'
    
    Returns:
        sequence_feature  -- padding 后的特征
    """

    def split(x, sep=','):
        """处理多值特征,该函数主要做分离"""

        if not isinstance(x, str):
            x = str(x)
        key_ans = x.split(sep)
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    feature_list = list(map(split, data[feature].values))    # 分割
    max_length = max_feature_length if max_feature_length else max(list(map(len, feature_list)))
    # padding 对齐
    padding_feature = padding_func(feature_list, max_length)
    dim = sequence_dim if sequence_dim else len(key2index) + 1
    sequence_feature = VarLenFeat(feature, dim, max_length, combiner, hash_flag=hashing, dtype="string")
    return sequence_feature


def sparse_feat_list_gen(data, sparse_features, mult=1, hashing=False):
    return [SingleFeat(feat, data[feat].nunique() * mult, hash_flag=hashing, dtype='string') for feat in sparse_features]


def dense_feat_list_gen(data, dense_features, hashing=False):
    return [SingleFeat(feat, 0, hash_flag=hashing, dtype='string') for feat in dense_features]
