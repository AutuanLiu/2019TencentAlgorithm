## 腾讯广告算法大赛

### 任务栏

|task|进度|完成日期|备注|
|---|---|---|---|
|数据预处理|20%|||


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
└── test_A -> data 的软连接路径

```

### 依赖

|name|细节|备注|
|---|---|---|
|modin|加速表格文件读取|256x|
|yapf|格式化文件|`yapf -i -r .`|
