# Part 2：云端优化

主程序保留题目的 `CloudOptimizer`、四项优化方法和 `benchmark_cloud_optimizations()`。使用已有的 WSL / Python 3.11 / TensorFlow 2.15.1 环境，依赖见 `requirements.txt`。

## 运行

在提交目录执行：

```bash
source ~/.venvs/maie5532/bin/activate
python part2/part2_cloud_optimization.py
```

也可以只运行一组：

```bash
python part2/part2_cloud_optimization.py mixed
python part2/part2_cloud_optimization.py distributed
python part2/part2_cloud_optimization.py batch
python part2/part2_cloud_optimization.py distillation
```

轮数等设置在文件顶部：`EPOCHS=50`、`TEMPERATURE=4.0`、`ALPHA=0.5`。试跑时将 `EPOCHS` 临时改成 2，通过后恢复。旧版 `--experiment`、`--smoke` 等参数已移除。

## 必要实验

| 实验 | 对照 |
|---|---|
| 混合精度 | 相同初始权重的 Float32 和混合精度；输出 Float32，动态损失缩放 |
| 分布式模拟 | 相同逻辑显存划分下的一副本和两副本，固定全局 batch 128 |
| 数据管线 | batch 128 下的串行映射与并行映射/预取 |
| 批大小 | batch 128 和 256 的吞吐比较 |
| 梯度累积 | 微批 64 累积四次，与直接 batch 256 比较 |
| 知识蒸馏 | 628378 参数的教师、324394 参数的普通学生和蒸馏学生 |

总计九种配置，普通 Float32 模型同时作为多个实验的对照。所有学生从相同随机初始权重训练，数据划分与 Part 1 一致。教师参数约为基线的 1.94 倍。

每个配置在独立子进程执行，以隔离精度策略、逻辑设备配置和显存统计。训练过程直接显示在终端；遇到错误停止并显示异常。

## 输出

固定保存在 `part2/part2_results/`：

- `cloud_optimized_models/<配置>/best_model.keras`：验证损失最低的普通网络。
- 同目录下的 `metrics.json` 和 `history.json`：指标及训练历史。
- `cloud_optimization_report.json`：本次所选实验组的汇总和成对比较。
- `performance_comparison.png`：准确率与训练吞吐对比图。

重跑会覆盖对应配置及汇总，需保留的正式结果请先备份。各组分别运行时，模型目录会保留其他组，但汇总只包含最近成功运行的组；一次运行全部组可得到完整报告。若运行失败，不要把旧汇总当成本次结果。旧版 `runs/` 下的已有实验不改动。

## 分析口径

- 本次各配置使用固定且一致的轮数及学习率时间表，与 Part 1 的自适应调度不同，故使用本次 Float32 作为直接对照。
- 总训练耗时包含验证和模型保存；稳态训练吞吐排除第一轮编译预热、验证和保存。推理平均延迟采用单张输入，包含输出回传，不含预处理和加载。
- 内存是 TensorFlow 分配器的字节数，不是进程总显存；逻辑设备的统计不能相加。模型导出不含优化器，文件大小不能直接与含优化器的 Part 1 文件比较。
- 模拟副本共享同一物理 GPU，效率不代表真实多卡扩展能力。普通 BatchNorm 仍按各副本/微批统计，累积梯度不保证与大批次训练轨迹完全一致。
- 蒸馏损失为标签交叉熵与温度 KL 散度的加权和，验证损失仅为学生交叉熵；教师保持冻结。分析总成本时需同时考虑教师训练。
- 保留 TensorFlow 默认 TF32 设置。GPU 数值计算与并行增强不保证逐位复现；优化是否有效应根据正式结果判断。

程序提供报告所需的数据与图表，书面分析在正式实验后编写。开发时的数值检查已完成，运行作业无需单独测试脚本。
