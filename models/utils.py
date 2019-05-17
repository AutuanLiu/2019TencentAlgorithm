import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from deepctr.utils import SingleFeat, VarLenFeat


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



def single_multi_value_feature_encoding(data, feature, padding_func, sequence_dim=None, max_feature_length=None, **kwargs):
    """单个特征操作
    
    Arguments:
        data (pd.DataFrame): -- 原始数据
        feature (str): -- 多值特征名
        padding_func (function) -- padding 函数
        max_feature_length (int) -- 自定义长度
        sequence_dim (int) -- 序列特征维度
    
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
    max_length = max_feature_length if max_feature_length else max(list(map(len, feature_list)))
    # padding 对齐
    padding_feature = padding_func(feature_list, max_length)
    dim = sequence_dim if sequence_dim else emb_sz_rule(len(key2index) + 1)
    print(f'len_unique： {len(key2index): 8d}, emb_sz： {dim: 8d}, max_len： {max_length: 8d}')
    del key2index
    sequence_feature = VarLenFeat(feature, dim, max_length, dtype="float32", **kwargs)
    return sequence_feature, padding_feature


def sparse_feat_list_gen(data, sparse_features, mult=1, hash_flag=False):
    return [SingleFeat(feat, data[feat].nunique() * mult, hash_flag=hash_flag, dtype='float32') for feat in sparse_features]


def dense_feat_list_gen(data, dense_features, hash_flag=False):
    return [SingleFeat(feat, 0, hash_flag=hash_flag, dtype='float32') for feat in dense_features]
