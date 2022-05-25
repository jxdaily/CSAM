# CSA

code for: " 基于输入通道拆分的对抗攻击迁移性增强算法" (计算机工程)

## 环境

- nvidia-tensorflow==1.15.4

- python==3.6
- nump==1.18
- scipy==1.2.1

## 实验

### 数据集

下载 [数据集](https://drive.google.com/open?id=1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) ，并放在 [dev_data/](https://github.com/JHL-HUST/SI-NI-FGSM/blob/master/dev_data)

### 预训练模型

下载预训练模型，并放在 [models/](https://github.com/JHL-HUST/SI-NI-FGSM/blob/master/models)

### 生成对抗样本

`python csa.py`

### 验证

`python eval.py`

## 致谢

代码参考：[SI-NI-FGSM]([JHL-HUST/SI-NI-FGSM (github.com)](https://github.com/JHL-HUST/SI-NI-FGSM))

