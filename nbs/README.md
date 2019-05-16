## 顺序

1. 数据信息查看与微处理[pro_data.ipynb](./pro_data.ipynb)
2. 广告静态特征数据预处理[data_clean.ipynb](./data_clean.ipynb)
3. 广告静态特征与广告操作合并[ad_operation_clean.ipynb](./ad_operation_clean.ipynb)
4. 曝光日志数据处理[log_clean.ipynb](./log_clean.ipynb)
5. ？[operation_to_ad.ipynb](./operation_to_ad.ipynb)
6. 用于构建标签数据，即每日的广告曝光量[label_construct.ipynb](./label_construct.ipynb)
7. 创建操作数据处理[train_ads_features_crt.ipynb](./train_ads_features_crt.ipynb)
8. 修改操作数据处理[train_ads_features-fix.ipynb](./train_ads_features-fix.ipynb)
9. 测试集特征处理[test_features.ipynb](./test_features.ipynb)
10. 初始数据集制作与打标签[train_test_set.ipynb](./train_test_set.ipynb)
11. 数据集后处理[post_pre.ipynb](./post_pre.ipynb)
12. 结果生成与提交[submit.ipynb](./submit.ipynb)

## python script order

1. 精简数据[log_reduced.py](./log_reduced.py)
2. 日志文件处理[log_clean.py](./log_clean.py)
3. 删除日志完全重复数据[log_duplicate_rm.py](./log_duplicate_rm.py)
4. 统计曝光量[sta_exposed.py](./sta_exposed.py)
5. 工具箱[utils.py](./utils.py)


## 注意事项

1. 使用数据前请**备份**
2. 运行 [log_clean.py](./log_clean.py)前请 **备份**，由于是增量修改，所以运行该文件会删除且覆盖原有的数据文件
