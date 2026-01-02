# 集中式神经网络模型 - Fashion-MNIST

这是一个基于Fashion-MNIST数据集的集中式神经网络实现，用作联邦学习模型的基准。

## 项目结构

```
Centralised/
├── data_loader.py      # 数据加载和预处理
├── model.py            # 神经网络模型定义
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── requirements.txt    # 依赖包
└── README.md          # 本文件
```

## 模型架构

### SimpleCNN (推荐)
- 卷积层1: 1 → 32 通道 (3×3 kernel)
- 最大池化: 2×2
- 卷积层2: 32 → 64 通道 (3×3 kernel)
- 最大池化: 2×2
- 全连接层1: 64×7×7 → 128
- 全连接层2: 128 → 10 (输出)
- Dropout: 0.5

### SimpleNN (可选)
- 全连接层1: 784 → 256
- 全连接层2: 256 → 128
- 全连接层3: 128 → 10 (输出)
- Dropout: 0.3

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练参数可在 `train.py` 中的 `main()` 函数中修改：
- `BATCH_SIZE`: 批大小 (默认: 32)
- `EPOCHS`: 训练轮数 (默认: 20)
- `LEARNING_RATE`: 学习率 (默认: 0.001)

训练过程中会：
- 保存最佳模型到 `logs/` 目录
- 记录训练历史到 `history.json`
- 生成TensorBoard日志

### 2. 评估模型

```bash
python evaluate.py
```

评估脚本会计算以下指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵

## 数据集

Fashion-MNIST数据集包含：
- 训练集: 60,000张图像
- 测试集: 10,000张图像
- 图像大小: 28×28 像素
- 类别数: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## 预期性能

使用SimpleCNN模型，预期可达到：
- 测试准确率: 90%+ 
- 训练时间: ~5-10分钟 (GPU) / ~30-60分钟 (CPU)

## 输出文件

训练完成后，会在 `logs/` 目录下生成：
- `best_model.pth`: 最佳模型权重
- `history.json`: 训练历史数据
- TensorBoard事件文件

## 查看TensorBoard

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开 `http://localhost:6006`

## 注意事项

1. 确保Fashion-MNIST CSV文件路径正确
2. 如果没有GPU，训练会自动使用CPU（速度较慢）
3. 第一次运行时会进行数据预处理，可能需要一些时间
4. 建议使用GPU加速训练过程

## 后续步骤

完成集中式模型训练后，可以：
1. 记录最终的准确率作为基准
2. 使用相同的数据集开发联邦学习模型
3. 比较两种方法的性能差异
