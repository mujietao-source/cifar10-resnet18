# CIFAR-10 图像分类实验

使用 PyTorch 实现 CIFAR-10 图像分类实验，包含数据处理、ResNet18 模型、训练、评估和结果可视化。

## 实验结果

提交版记录的最佳验证准确率为 **91.26%**，测试准确率为 **90.38%**，宏平均 F1 为 **90.43%**。这些数字来自本次实验记录，重跑结果可能随环境和随机性变化。

结果图位于 `outputs/final/`，包括准确率曲线、损失曲线和混淆矩阵。模型权重和 CIFAR-10 数据集未上传。

## 运行

```bash
python -m pip install -r requirements.txt
python train.py --help
python evaluate.py --help
```

主要文件：`data.py` 负责数据划分与增强，`model.py` 定义模型，`train.py` 训练，`evaluate.py` 计算指标，`plot_history.py` 生成曲线。
