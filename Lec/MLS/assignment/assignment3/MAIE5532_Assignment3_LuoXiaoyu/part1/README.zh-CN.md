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

第一阶段最多训练 8 个 epoch，学习率为 `1e-3`。第二阶段最多训练 25 个
epoch，学习率为 `1e-4`。两个阶段都会保存逐 epoch 日志，并在结束时恢复
验证准确率最佳的权重。

## 输出

| 路径 | 内容 |
|---|---|
| `baseline_mobilenetv2.keras` | 最终训练模型 |
| `part1_results/training_stage1_log.csv` | 冻结主干阶段的训练日志 |
| `part1_results/training_stage2_log.csv` | 微调阶段的训练日志 |
| `part1_results/baseline_metrics.json` | 准确率和性能指标 |

精简后的流程只生成上述文件。独立的 checkpoint、训练历史 JSON 和训练曲线
图片不是第 1 部分的强制交付物，因此不会生成。

## 已记录结果

| 指标 | 结果 |
|---|---:|
| CIFAR-10 测试准确率 | 93.71% |
| CIFAR-10 测试损失 | 0.2242 |
| 参数量 | 2,270,794 |
| 单张图像 FLOPs | 599,594,364 |
| `.keras` 模型文件大小 | 26.0085 MiB |
| 模型权重内存 | 8.6624 MiB |
| 单样本平均延迟 | 2.4972 ms |
| Batch 32 平均延迟 | 11.8789 ms |

第一阶段完成 8 个 epoch，最佳验证准确率为 81.08%。第二阶段完成 25 个
epoch，在最后一个 epoch 达到最佳验证准确率 94.02% 和最佳验证损失
0.2051。最终模型使用第二阶段验证准确率最佳的权重。

26.0085 MiB 的 `.keras` 文件包含模型结构和序列化的训练状态；8.6624 MiB
只统计 TensorFlow 模型权重和 BatchNorm 状态。这两个数字的测量对象不同。

延迟包含 TensorFlow 前向计算和输出传回主机的过程，不包含图像缩放、预处理、
模型加载和磁盘访问。单样本结果在预热后测量 100 次，Batch 32 结果在预热后
测量 40 次。由于没有可靠的逐推理能耗计数器，能耗未记录。
