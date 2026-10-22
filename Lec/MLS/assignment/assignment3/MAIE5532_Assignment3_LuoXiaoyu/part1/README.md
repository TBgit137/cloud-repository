# 第 1 部分：MobileNetV2 基线

本目录实现 CIFAR-10 的 MobileNetV2 迁移学习基线，并记录单样本、批量推理、模型大小、内存、参数量和 FLOPs。

## 环境

推荐使用 assignment 2 已有的 WSL 环境：

```bash
cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

首次训练需要联网下载 MobileNetV2 的 ImageNet 权重。CIFAR-10 已存在缓存时不会重新下载。

## 完整运行

```bash
python part1/part1_baseline_model.py
```

默认设置：

- 训练批大小：64
- 验证集比例：10%
- 第一阶段：最多 8 个 epoch，基础模型冻结
- 第二阶段：最多 25 个 epoch，以 `1e-4` 学习率微调
- 基准测试：单样本 100 次，批大小 32 的推理 40 次

快速测试参数解析和模型构建时可以缩短训练，但短训练结果不应作为最终提交结果：

```bash
python part1/part1_baseline_model.py \
  --stage1-epochs 1 \
  --stage2-epochs 1 \
  --inference-runs 10 \
  --batch-inference-runs 5
```

只训练而不执行性能基准测试：

```bash
python part1/part1_baseline_model.py --no-benchmark
```

对已保存的模型重新执行性能和准确率测试，不重新训练：

```bash
python part1/part1_baseline_model.py --benchmark-only
```

若训练已经超过 85% 目标并手动停止，可以使用第二阶段验证集最佳 checkpoint 补齐最终模型、训练图表和性能指标：

```bash
python part1/part1_baseline_model.py --finalize-from-checkpoint
```

## 输出

| 路径 | 内容 |
|---|---|
| `baseline_mobilenetv2.keras` | 最终训练模型 |
| `part1_results/training_stage1_log.csv` | 冻结基础模型阶段的逐 epoch 日志 |
| `part1_results/training_stage2_log.csv` | 微调阶段的逐 epoch 日志 |
| `part1_results/training_history.json` | 两阶段完整训练历史 |
| `part1_results/training_manifest.json` | 实际训练参数、完成 epoch 数和训练耗时 |
| `part1_results/training_curves.png` | 训练和验证曲线，标出微调起点 |
| `part1_results/model_summary.txt` | 模型结构及参数量 |
| `part1_results/baseline_metrics.json` | 准确率、延迟、内存、模型大小、FLOPs 和运行环境 |
| `part1_results/stage1_best.keras` | 第一阶段验证集最佳 checkpoint |
| `part1_results/stage2_best.keras` | 第二阶段验证集最佳 checkpoint |

若测试准确率没有严格超过 85%，程序会在写出全部诊断结果后以非零状态退出，便于识别未达到验收目标。

## 本次参考运行

本机使用 WSL2 和 NVIDIA GeForce RTX 4070 Ti 运行。训练在目标已明显超过后手动结束，并从第二阶段验证集最佳 checkpoint 生成最终模型：

| 指标 | 结果 |
|---|---:|
| CIFAR-10 测试准确率 | 93.71% |
| 参数量 | 2,270,794 |
| FLOPs（单张图像） | 599,594,364 |
| 模型文件大小 | 9.16 MiB |
| 单样本延迟 | 2.289 ms（均值） |
| batch size 32 延迟 | 11.114 ms（均值） |
| batch size 32 吞吐量 | 2,879 images/s |
| GPU 分配器峰值 | 616.5 MiB |

单样本延迟的 P95 为 3.22 ms，中位数为 2.01 ms。不同次运行会受 WSL 调度影响，实际提交以 `part1_results/baseline_metrics.json` 中当前那次测量的完整记录为准。能耗未作为实测指标记录，因为 WSL 中没有可靠的逐推理能耗计数器。
