## 腾讯广告算法大赛

### 任务栏

|task|进度|完成日期|备注|
|---|---|---|---|
|数据清洗|100%|2019.5.9||
|训练集制作|70%|||


### 进度描述
#### 1. 数据清洗
**1.1 ad_static_features**
1. 去掉存在缺失值/创建时间为0的记录
2. 去掉日期异常的记录
3. 多值处理（商品id，广告行业wid存在多值， 需要清理）

**1.2 imps_out**
1. 去掉存在空值的记录
2. imps_out广告id必须在ad_stastic_feature中存在，否则异常
3. 去掉日期异常的记录
4. 去掉重复的记录
5. 去掉数据类型不符合的数据

**1.3 ad_operations**
1. 去掉存在空值的记录
2. 去掉ad_stastic_feature中不存在的广告，否则异常
3. 去掉日期异常的记录
4. 重复操作的记录

**1.4 user**
暂不需清洗

#### 2.训练集制作
**2.1 根据ad_operation操作ad_static**
1. 操作类型1，初始化属性值
2. 操作类型2，更改属性值，建立新的纪录
3. 删除只有更改操作，没有新建操作的纪录
4. 广告状态？ 暂不删除

**2.2 统计广告id的日曝光值，制作label**


#### 3.特征工程

#### 4.模型训练



### 目录树

```bash
tree .

.
├── main.py
├── models
│   ├── layers.py
│   ├── loss.py
│   ├── metrics.py
│   └── network.py
├── nbs
│   ├── baseline.ipynb
│   └── pro_data.ipynb
├── README.md
├── requirements.txt
└── test_A -> data 的软连接路径(快捷方式)

```

### 依赖

|name|细节|备注|
|---|---|---|
|modin|加速表格文件读取|256x|
|yapf|格式化文件|`yapf -i -r .`|
