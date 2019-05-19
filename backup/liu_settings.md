| 时间                     | 模型    | 模型设置                                                                                                                                                                                                                                 | 学习器                   | 学习器设置                                                             | 其它有效设置    | 验证损失 | 备份文件名                                        |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------- | --------------- | -------- | ------------------------------------------------- |
| Sun-May-19-19:17:36-2019 | xDeepFM | `model_settings = {"embedding_size": 8, "dnn_use_bn": True, "dnn_dropout": 0.0}`                                                                                                                                                         | adabound, early_stopping | `AdaBound(lr=5e-6, final_lr=1e-3, weight_decay=0.001, amsbound=False)` | `emb_rule=True` | 0.6287   | Sun-May-19-19:17:36-2019_submission.csv           |
| 2019-05-18 18:41         | FGCNN   | `model_settings = {"embedding_size": 8,"conv_kernel_width": (3, 3, 3),"conv_filters": (5, 8, 5),"new_maps": (3, 3, 3),"pooling_width": (2, 2, 2),"dnn_hidden_units": (128, ),"l2_reg_embedding": 1e-5,"l2_reg_dnn": 0,"dnn_dropout": 0}` | adabound, early_stopping | `AdaBound(lr=5e-4, final_lr=1e-1, weight_decay=0.001, amsbound=False)` | `emb_rule=True` | 0.4337   | 449af160b1b636196a61b80e8599ca74_submission.csv， |



- 计算 sha1sum

```bash
$ sha1sum submission.csv 
```

- 计算 md5sum

```bash
$ md5sum submission.csv 

305c80b5d95ca48957c17c2d488390cf
```