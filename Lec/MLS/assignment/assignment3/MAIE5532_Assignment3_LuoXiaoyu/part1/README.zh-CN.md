# 第 1 部分：MobileNetV2 基线（中文版）

本目录包含第 1 部分要求的 CIFAR-10 基线，使用 ImageNet 预训练的
MobileNetV2、自定义分类头和两阶段迁移学习。

## 环境

本次记录运行使用 Python 3.11.16、TensorFlow 2.15.1、Keras 2.15.0、
WSL2 Ubuntu 和 NVIDIA GeForce RTX 4070 Ti。

在 `MAIE5532_Assignment3_LuoXiaoyu` 目录中执行：

```bash
python3.11 -m venv ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

## 运行

```bash
python part1/part1_baseline_model.py
```

脚本先冻结 MobileNetV2 并训练分类头，再解冻非 BatchNorm 层并以较低学习率
微调。最终模型会在全部 10,000 张 CIFAR-10 测试图像上评估，并测量单样本和
批量推理延迟。

## 输出

| 路径 | 内容 |
|---|---|
| `baseline_mobilenetv2.keras` | 最终训练模型 |
| `part1_results/training_stage1_log.csv` | 冻结主干阶段的训练日志 |
| `part1_results/training_stage2_log.csv` | 微调阶段的训练日志 |
| `part1_results/baseline_metrics.json` | 准确率和性能指标 |

## 已记录结果

| 指标 | 结果 |
|---|---:|
| CIFAR-10 测试准确率 | 93.71% |
| 参数量 | 2,270,794 |
| 单张图像 FLOPs | 599,594,364 |
| 模型大小 | 9.1617 MiB |
| 单样本平均延迟 | 2.2894 ms |
| Batch 32 平均延迟 | 11.1140 ms |
| 模型权重内存 | 8.6624 MiB |

第一阶段完成 4 个 epoch，第二阶段完成 14 个 epoch。最终模型使用微调阶段
验证准确率最佳时的权重。

延迟包含 TensorFlow 前向计算和输出传回主机的过程，不包含图像缩放、预处理、
模型加载和磁盘访问。由于没有可靠的逐推理能耗计数器，能耗记录为 `null`。
