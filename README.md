# 腾讯广告算法大赛

## 目录

- [腾讯广告算法大赛](#%E8%85%BE%E8%AE%AF%E5%B9%BF%E5%91%8A%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B)
  - [目录](#%E7%9B%AE%E5%BD%95)
  - [1. 任务栏](#1-%E4%BB%BB%E5%8A%A1%E6%A0%8F)
  - [2. 数据处理](#2-%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86)
    - [2.1 广告静态特征数据处理](#21-%E5%B9%BF%E5%91%8A%E9%9D%99%E6%80%81%E7%89%B9%E5%BE%81%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86)
    - [2.2 广告操作数据处理](#22-%E5%B9%BF%E5%91%8A%E6%93%8D%E4%BD%9C%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86)
    - [2.3 曝光日志数据处理](#23-%E6%9B%9D%E5%85%89%E6%97%A5%E5%BF%97%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86)
    - [2.4 每日曝光量统计](#24-%E6%AF%8F%E6%97%A5%E6%9B%9D%E5%85%89%E9%87%8F%E7%BB%9F%E8%AE%A1)
  - [3.数据集制作](#3%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%B6%E4%BD%9C)
    - [3.1 根据操作数据提取有效特征](#31-%E6%A0%B9%E6%8D%AE%E6%93%8D%E4%BD%9C%E6%95%B0%E6%8D%AE%E6%8F%90%E5%8F%96%E6%9C%89%E6%95%88%E7%89%B9%E5%BE%81)
    - [3.2 测试集特征处理](#32-%E6%B5%8B%E8%AF%95%E9%9B%86%E7%89%B9%E5%BE%81%E5%A4%84%E7%90%86)
    - [3.3 训练集特征处理](#33-%E8%AE%AD%E7%BB%83%E9%9B%86%E7%89%B9%E5%BE%81%E5%A4%84%E7%90%86)
    - [3.4 后处理](#34-%E5%90%8E%E5%A4%84%E7%90%86)
  - [4.模型训练](#4%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
    - [4.1 DeepFM](#41-deepfm)
    - [4.2 xDeepFM](#42-xdeepfm)
  - [5. 结果提交](#5-%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4)
  - [6. 目录树](#6-%E7%9B%AE%E5%BD%95%E6%A0%91)
  - [7. 依赖](#7-%E4%BE%9D%E8%B5%96)
  - [8. 软链接的创建方式](#8-%E8%BD%AF%E9%93%BE%E6%8E%A5%E7%9A%84%E5%88%9B%E5%BB%BA%E6%96%B9%E5%BC%8F)

## 1. 任务栏

| task       | 进度 | 完成日期  | 备注 |
| ---------- | ---- | --------- | ---- |
| 数据预处理 | 100% | 2019.5.9  |      |
| 数据清洗   | 100% | 2019.5.9  |      |
| 数据集制作 | 100% | 2019.5.16 |      |
| 模型训练   |      |           |      |


## 2. 数据处理

### 2.1 广告静态特征数据处理

1. 去掉存在缺失值/创建时间为0的记录或者创建时间无效记录，如 `2019/2/29` 之类的
2. 多值处理（商品id，广告行业id存在多值， 需要清理）不存在该种异常值
3. 将日期转化为 `datetime` 类型
4. 删除创建时间缺失的广告（这部分应该放在最后，这里提前处理会有问题）
5. 输出 `ad_static.csv`

### 2.2 广告操作数据处理

1. 去掉存在空值的记录
2. imps_out广告id必须在 `ad_stastic_feature` 中存在，否则异常，通过 `inner` 合并来解决
3. 去掉日期异常的记录
4. 去掉重复的记录
5. 去掉数据类型不符合的数据
6. 如果广告操作数据的修改时间为0，则其为广告的创建时间，赋值操作
7. 输出 `ad_statics_operation.csv`

### 2.3 曝光日志数据处理

1. `广告请求时间` 转化为 datetime
2. 去掉存在空值的记录， 去掉日期异常的记录
3. 新建 `广告请求时间_date` 列， 用于按时间分类 如 2019-2-18
4. 去掉 `ad_stastic_feature` 中不存在的广告，`inner` 合并
5. 由于是追加的方式保存文件，所以先判断工作环境的安全
6. 逐行保存文件到对应日期的文件中
7. 因为只提取想要的数据，所以 生成一个 `log_reduced.csv` 保存精简信息
8. 输出 2019-2-18_log.csv 等文件 32 个
9. 重新读入文件 加 header 和 去掉重复操作的记录， 保存为 同名文件

### 2.4 每日曝光量统计

1. `log.groupby(by=['曝光广告id']).count()` 统计曝光量并保存


## 3.数据集制作

### 3.1 根据操作数据提取有效特征

1. 根据 `操作类型` 将 `ad_statics_operation` 分割为新建和修改两部分
2. 空值使用 -999 填充
3. 分别操作新建和修改两部分数据
   1. 将新建和修改数据按照`广告id`进行排序并统计每个广告的操作次数保存备用
   2. 创建操作是在同一个广告id上不断叠加的结果 28685条
   3. 增加定向人群相关特征
   4. 将创建时间更改为最后的修改时间
   5. 修改操作是继承上一条特征的基础上修改字段，同样使用统计思想简化，保留原始长度
   6. 删除失效状态广告

### 3.2 测试集特征处理

1. 人群定向特征分解
2. -9 表示特征取值不限， -999表示特征缺失
3. 新增特征 `has_product_id` 表示是否有产品id，代表不同的广告类型
4. 新增特征 `is_all_field` 表示人群定向特征是否是全部
5. 新增日期相关特征
6. 数据类型转换

### 3.3 训练集特征处理

1. 删除3月19日创建的广告，因为没有标签
2. 训练集打标签： 2月16号以前创建的广告，认为是静态广告，设置不变，不参与竞价，从2月16号到3月19号每天都有标签，2月16号之后创建的数据认为是动态广告，只有后一天才存在标签（可能有问题）
3. 解决方案
    1. 时间列 排序
    2. 统计每种时间的个数
    3. 按索引取出对应的时间子表
    4. 读取标签并 merge
4. 人群定向特征分解
5. 添加时间特征

### 3.4 后处理

1. 统一特征、数据类型、数值表示
2. 重排特征顺序
3. 连续特征做对数变换


## 4.模型训练

### 4.1 DeepFM

### 4.2 xDeepFM

## 5. 结果提交

1. 预测结果取指数变换
2. 结果生成与提交


## 6. 目录树

```bash
tree .

.
├── data
│   ├── 2019-02-16_log.csv
│   ├── 2019-02-16_log_exposed.csv
│   ├── 2019-02-17_log.csv
│   ├── 2019-02-17_log_exposed.csv
│   ├── 2019-02-18_log.csv
│   ├── 2019-02-18_log_exposed.csv
│   ├── 2019-02-19_log.csv
│   ├── 2019-02-19_log_exposed.csv
│   ├── 2019-02-20_log.csv
│   ├── 2019-02-20_log_exposed.csv
│   ├── 2019-02-21_log.csv
│   ├── 2019-02-21_log_exposed.csv
│   ├── 2019-02-22_log.csv
│   ├── 2019-02-22_log_exposed.csv
│   ├── 2019-02-23_log.csv
│   ├── 2019-02-23_log_exposed.csv
│   ├── 2019-02-24_log.csv
│   ├── 2019-02-24_log_exposed.csv
│   ├── 2019-02-25_log.csv
│   ├── 2019-02-25_log_exposed.csv
│   ├── 2019-02-26_log.csv
│   ├── 2019-02-26_log_exposed.csv
│   ├── 2019-02-27_log.csv
│   ├── 2019-02-27_log_exposed.csv
│   ├── 2019-02-28_log.csv
│   ├── 2019-02-28_log_exposed.csv
│   ├── 2019-03-01_log.csv
│   ├── 2019-03-01_log_exposed.csv
│   ├── 2019-03-02_log.csv
│   ├── 2019-03-02_log_exposed.csv
│   ├── 2019-03-03_log.csv
│   ├── 2019-03-03_log_exposed.csv
│   ├── 2019-03-04_log.csv
│   ├── 2019-03-04_log_exposed.csv
│   ├── 2019-03-05_log.csv
│   ├── 2019-03-05_log_exposed.csv
│   ├── 2019-03-06_log.csv
│   ├── 2019-03-06_log_exposed.csv
│   ├── 2019-03-07_log.csv
│   ├── 2019-03-07_log_exposed.csv
│   ├── 2019-03-08_log.csv
│   ├── 2019-03-08_log_exposed.csv
│   ├── 2019-03-09_log.csv
│   ├── 2019-03-09_log_exposed.csv
│   ├── 2019-03-10_log.csv
│   ├── 2019-03-10_log_exposed.csv
│   ├── 2019-03-11_log.csv
│   ├── 2019-03-11_log_exposed.csv
│   ├── 2019-03-12_log.csv
│   ├── 2019-03-12_log_exposed.csv
│   ├── 2019-03-13_log.csv
│   ├── 2019-03-13_log_exposed.csv
│   ├── 2019-03-14_log.csv
│   ├── 2019-03-14_log_exposed.csv
│   ├── 2019-03-15_log.csv
│   ├── 2019-03-15_log_exposed.csv
│   ├── 2019-03-16_log.csv
│   ├── 2019-03-16_log_exposed.csv
│   ├── 2019-03-17_log.csv
│   ├── 2019-03-17_log_exposed.csv
│   ├── 2019-03-18_log.csv
│   ├── 2019-03-18_log_exposed.csv
│   ├── 2019-03-19_log.csv
│   ├── 2019-03-19_log_exposed.csv
│   ├── ads_crt.csv
│   ├── ads_fix1.csv
│   ├── ads_fix2.csv
│   ├── ads_fix.csv
│   ├── ad_static_all.csv
│   ├── ad_static.csv
│   ├── ad_static_operation.csv
│   ├── log_data.tgz
│   ├── log_label.tgz
│   ├── log_reduced.csv
│   ├── sta_log.csv.sh
│   ├── submission.csv
│   ├── test_data.csv
│   ├── test_set.csv
│   ├── test_set_final.csv
│   ├── train_set.csv
│   ├── train_set_final.csv
│   └── users_data.csv
├── info_FAQ.md
├── main.py
├── models
│   ├── adabound.py
│   ├── adamw.py
│   ├── clr_callback.py
│   ├── __init__.py
│   ├── layers.py
│   ├── loss.py
│   ├── metrics.py
│   ├── network.py
│   └── utils.py
├── nbs
│   ├── add_timestamp.ipynb
│   ├── ad_operation_clean.ipynb
│   ├── data_clean.ipynb
│   ├── label_construct.ipynb
│   ├── log_clean.ipynb
│   ├── log_clean.py
│   ├── log_duplicate_rm.py
│   ├── log_reduced.py
│   ├── operation_to_ad.ipynb
│   ├── post_pre.ipynb
│   ├── pro_data.ipynb
│   ├── __pycache__
│   │   └── utils.cpython-36.pyc
│   ├── README.md
│   ├── sta_exposed.py
│   ├── submit.ipynb
│   ├── test_features.ipynb
│   ├── train_ads_features_crt.ipynb
│   ├── train_ads_features-fix.ipynb
│   ├── train_test_set.ipynb
│   └── utils.py
├── README.md
├── requirements.txt
├── test_A -> /home/lab/Datasets/Tencent/test_A/
└── TrainingModel.ipynb

7 directories, 121 files

```

## 7. 依赖

| name   | 细节               | 备注           |
| ------ | ------------------ | -------------- |
| modin  | 加速表格文件读取   | 256x           |
| yapf   | 格式化文件         | `yapf -i -r .` |
| fastai | 模型快速构建与训练 |                |

## 8. 软链接的创建方式

1. 解决数据问题

```bash
$ ln -s src dest
```
