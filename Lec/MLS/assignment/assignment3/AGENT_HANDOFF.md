# MLS Assignment 3 Agent 交接文档

最后更新：2026-10-22（Asia/Shanghai）

## 0. 接手须知

这是一份给后续 Codex/deepseek agent 的项目交接文档。接手后先阅读本文件，再检查当前工作区中的实际文件。不要只依赖本文件中的旧路径或结果摘要。

项目根目录：

```text
D:\work\cloud-repository\Lec\MLS
```

assignment 3 目录：

```text
D:\work\cloud-repository\Lec\MLS\assignment\assignment3
```

当前 active 提交目录：

```text
D:\work\cloud-repository\Lec\MLS\assignment\assignment3\MAIE5532_Assignment3_LuoXiaoyu
```

当前工作树状态：

- 整个 `assignment/assignment3/` 在 Git 中仍是未跟踪目录，状态为 `?? assignment/assignment3/`。
- 尚未创建 commit。
- 不要擅自清理、覆盖或重新生成模型文件；这些文件是 Part 1 的实际交付结果。

## 1. 用户目标与当前进度

课程作业是 **Hardware-Aware Design Assignment**，满分 100 分：

- Part 1：基线模型，25 分。
- Part 2：硬件感知优化，30 分。
- Part 3：多平台分析，25 分。
- Part 4：综合分析与设计建议，20 分。

用户已经明确选择 **Track B: Simulation & Performance Modeling**，不选 Track A 的真实硬件部署。

当前进度：

- 已把原始 PDF/DOCX 作业要求转成英文 Markdown 和中文 Markdown。
- 已完成 Part 1 的代码、正式训练、模型保存、训练日志、训练曲线和性能指标。
- Part 1 测试准确率为 **93.71%**，超过作业要求的 85%。
- Part 2、Part 3、Part 4 尚未实现。

用户的工作节奏是“一步一步来”。当前下一步自然任务是 Part 2，但在修改 Part 2 前应再次核对实际用户指令。

## 2. 用户已经表达的风格和偏好

### 2.1 注释语言

- 代码注释和 docstring 当前全部使用中文，方便用户阅读。
- 作业要求中的非 `TODO` 说明性注释应保留。
- 已完成的 `TODO` 注释已删除。
- 最终准备提交给老师时，再统一把 Python 注释和 docstring 转成英文。
- 现在不要提前把注释改成英文。

### 2.2 作业完成标准

- 用户明确说过“不需要非常细微调到很完美，只要完成作业要求就可以了”。
- 不要为了多 0.1% 准确率或几毫秒延迟无限训练和调参。
- 优先保证要求、交付物、可复现性和结果可信。

### 2.3 沟通与交付

- 用户使用中文交流，回答应简洁、直接，说明实际完成了什么。
- 不要只写计划而不说明当前 blocker。
- 重要结果应保存到 JSON、CSV、PNG 或 Markdown，不能只在终端输出。
- 后续交付最终课程作业时，应再检查目录和文件命名。

## 3. 原始作业要求和已转换文档

原始文件：

- `assignment/assignment3/requirement.pdf`
- `assignment/assignment3/requirement.docx`

已生成：

- `assignment/assignment3/Hardware-Aware Design Assignment.md`
- `assignment/assignment3/Hardware-Aware Design Assignment-中文.md`

转换事实：

- PDF 共 19 页。
- PDF 与 DOCX 内容一致。
- 文档中没有图片、公式对象或需要特殊转录的表格。
- 代码共拆成 9 个 Python 代码块。
- 英文版与中文版各有 9 个代码块。
- 所有代码块通过 Python 语法检查。
- 中文版去注释和 docstring 后，代码语法树均与英文版一致。
- 中文版代码注释、TODO 提示和 docstring 均已翻译成中文。
- 转换时逐页检查过 19 页 PDF 渲染，没有遗漏的图或表。

## 4. Track A / Track B 决策记录

### 4.1 当时给用户的结论

- 若没有三种真实硬件，Track B 更稳妥。
- Track B 不需要购买或借用硬件，并且可复用 assignment 2 的 TFLite、性能分析和报告经验。
- 两条路线在作业说明中都可以拿满分。
- Track A 要求在至少 3 种不同的真实硬件平台类型上完成可运行部署、真实测量和压力测试。

### 4.2 “3 种平台类型”的解释

当时建议不要把“三个设备”简单等同于“三种平台类型”。合理类别包括：

- x86 CPU。
- ARM Cortex-A。
- 移动 GPU，例如 Adreno、Mali、Apple GPU。
- Cortex-M MCU。
- Edge AI 加速器，例如 Coral Edge TPU。

建议的安全组合：

- x86 CPU。
- ARM Cortex-A 或移动 GPU。
- Cortex-M 或 Edge TPU。

三台同架构 x86 机器或三个虚拟机通常不能算三种平台类型。ARM 模拟器也不能算 Track A 的真实硬件。

### 4.3 用户最终决定

用户明确说：“我还是老老实实走 trackb 吧。”

因此后续代码和报告统一按 Track B 编写，例如：

- `part3_modeling.py`，而不是 `part3_deployment.py`。
- 性能模型。
- QEMU ARM 验证。
- Renode/QEMU Cortex-M 验证。
- WebGPU/Vulkan 代理分析。
- 明确标记 `[native-x86]`、`[QEMU-ARM]`、`[Renode-CM]`、`[WebGPU-proxy]` 等执行环境。

## 5. 课程和既往作业上下文

相关课件：

- `course/MAIE 5532 - Week 6 - Hardware Backend Specialization and Optimizations.pdf`
- `course/MAIE 5532 - Week 7 - Machine Learning for Systems.pdf`
- `course/MAIE 5532 - Week 8 - Automating ML Compilation.pdf`

作业说明明确说它建立在第 6-8 周硬件加速内容上。

可复用既往作业：

- `assignment/assignment1/`
- `assignment/assignment2/MAIE5532_Assignment2_LuoXiaoyu/`

assignment 2 中已经存在：

- CIFAR-10 基线训练。
- TFLite 和 INT8 量化。
- 剪枝、轻量模型和 NAS。
- 多尺度部署报告。
- 性能测量和训练曲线。
- `part1/part1_baseline.py`
- `part2/part2_cloud_optimization.py`
- `part3/part3_edge_optimization.py`
- `part4/part4_deployment_pipeline.py`

这些脚本可以作为 Part 2-3 的实现模式来源，但不能直接假设 assignment 3 的模型、输入尺寸或要求完全相同。

仓库中没有找到 `AGENTS.md`。

## 6. Part 1 正式要求

Part 1 占 25 分，要求实现 MobileNetV2 + CIFAR-10 图像分类基线。

硬性要求：

- MobileNetV2 使用 ImageNet 预训练权重。
- 输入尺寸为 `224 x 224 x 3`。
- `include_top=False`。
- 添加自定义分类头。
- 分类头包含 Global Average Pooling 和 Dropout。
- 初始阶段冻结 MobileNetV2。
- 采用两阶段迁移学习：
  1. 冻结基础模型，只训练分类头。
  2. 解冻基础模型，以较低学习率微调。
- 加载 CIFAR-10。
- 缩放到 224×224。
- 做归一化和训练集数据增强。
- 使用 `tf.data.Dataset`。
- 保存 `baseline_mobilenetv2.keras`。
- 测量单样本和批推理延迟。
- 测量内存、模型大小、参数量和 FLOPs。
- 能耗“如果平台支持”则测量。
- CIFAR-10 测试准确率目标：`> 85%`。

官方交付物：

- 完整 MobileNetV2 基线实现。
- 展示迁移学习过程的训练日志。
- 测试准确率。
- 延迟、内存和模型大小等基线指标。

## 7. Part 1 当前代码

主程序：

```text
assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu/part1/part1_baseline_model.py
```

运行说明：

```text
assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu/part1/README.md
```

依赖：

```text
assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu/requirements.txt
```

主程序包含：

- 固定随机种子 `42`。
- GPU 显存增长配置。
- MobileNetV2 预训练模型。
- 模型内部 `Rescaling(1/127.5, offset=-1)`。
- GlobalAveragePooling2D。
- Dropout 0.2。
- Dense(10, softmax)。
- 训练集按类别分层验证集划分。
- 训练阶段在线增强和在线 resize。
- 两阶段训练。
- 第一阶段 checkpoint。
- 第二阶段 checkpoint。
- EarlyStopping。
- ReduceLROnPlateau。
- CSVLogger。
- TerminateOnNaN。
- 训练曲线。
- 同步延迟测量。
- 进程 RSS 记录。
- TensorFlow GPU 分配器内存记录。
- TensorFlow profiler FLOPs 估算。
- JSON 指标输出。
- 模型重新加载和 benchmark-only 模式。
- 手动中断后的 checkpoint 收尾模式。

代码规模约 1015 行，原因是把训练、回调、日志、评估、CLI 和异常收尾都放入了同一个可执行脚本。

## 8. Part 1 实现细节

### 8.1 数据和内存策略

- CIFAR-10 原始训练集共 50,000 张。
- 测试集 10,000 张。
- 验证集使用原始训练集的 10%，按类别分层划分。
- 实际约为 40,500 张训练图和 4,500 张验证图。
- 测试集不参与训练或 checkpoint 选择。
- 不把全部 224×224 图像预先展开成 NumPy 数组。
- 224×224 resize 在 `tf.data` 中动态完成，避免约 30 GB 的内存占用。

### 8.2 预处理

- CIFAR-10 加载后保持 0-255 浮点范围。
- 模型内部 Rescaling 映射到 `[-1, 1]`。
- 没有使用“先除以 255，再调用 MobileNetV2 preprocess_input”的重复缩放。

### 8.3 数据增强

训练集使用：

- 水平翻转。
- 轻度旋转 `factor=0.03`。
- 缩放 `height_factor=0.08, width_factor=0.08`。
- 平移 `height_factor=0.05, width_factor=0.05`。

增强先在 32×32 图像上执行，再 resize 到 224×224，目的之一是减少增强计算量。

验证和测试不做增强。

### 8.4 两阶段训练

第一阶段：

- 基础模型冻结。
- Adam，学习率 `1e-3`。
- 最大 8 epoch。
- EarlyStopping patience 3。
- ReduceLROnPlateau patience 2。
- checkpoint 按 `val_accuracy` 选择。

第二阶段：

- 从第一阶段最佳 checkpoint 继续。
- 解冻 MobileNetV2 非 BatchNorm 层。
- BatchNorm 层保持冻结。
- Adam，学习率 `1e-4`。
- 最大 25 epoch。
- EarlyStopping patience 6，监控 `val_loss`。
- ReduceLROnPlateau patience 3。
- checkpoint 按 `val_accuracy` 选择。
- 最终使用第二阶段最佳验证准确率 checkpoint。

### 8.5 性能测量口径

- 单样本延迟默认测量 100 次。
- batch size 32 延迟默认测量 40 次。
- 预热 20 次。
- 计时包含 TensorFlow 前向计算和输出传回 host。
- 计时不包含图像 resize、预处理、模型加载和磁盘访问。
- 每次调用后执行 `.numpy()` 以确保设备计算完成再停止计时。
- `memory_usage_mb` 当前是模型 TF 变量权重内存，包含 BatchNorm 状态。
- 额外记录进程 RSS 变化和 GPU 分配器 current/peak。
- `.keras` 文件大小单独记录。
- 不同内存口径不能混为一个数字。

## 9. Part 1 实际运行历史

### 9.1 冒烟测试

曾用 `--stage1-epochs 1 --stage2-epochs 1` 跑过完整 1+1 epoch 冒烟测试：

- 第一阶段验证准确率约 80.8%。
- 第二阶段验证准确率约 87.3%。
- 测试准确率约 87.15%。
- 冒烟测试生成的模型、日志和指标随后被删除，没有混入最终交付物。

### 9.2 正式训练

正式运行：

```bash
python part1/part1_baseline_model.py
```

第一阶段：

- 实际运行 4 epoch 后 EarlyStopping。
- 最佳验证准确率 `0.8084`，出现在第 1 epoch。

第二阶段：

- 实际完成并写入 CSV 的有 14 epoch。
- 最佳验证准确率 `0.9430`，出现在显示为 Epoch 11 的 checkpoint。
- 之后训练准确率继续上升，但验证指标波动，出现过拟合迹象。

用户说不需要继续精调后，训练被手动停止。随后使用：

```bash
python part1/part1_baseline_model.py --finalize-from-checkpoint
```

从 `stage2_best.keras` 和 CSV 日志补齐：

- 最终模型。
- 训练历史 JSON。
- 训练曲线。
- 训练 manifest。
- 最终性能指标。

`training_manifest.json` 中的事实：

- `run_mode`: `finalized_from_best_checkpoint`
- 配置上限：Stage 1 = 8，Stage 2 = 25。
- 实际完成：Stage 1 = 4，Stage 2 = 14。
- `training_seconds` 为 `null`，因为手动中断后没有保留完整训练墙钟时间。
- 有明确 `completion_note` 说明训练在达到目标后停止，并使用第二阶段最佳 checkpoint。

不要把这个模型误写成“完整跑满 8+25 epoch”；只能写成“配置上限 8+25，实际 4+14，使用最佳 checkpoint”。

## 10. Part 1 最终结果

最终模型：

```text
assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu/part1/baseline_mobilenetv2.keras
```

最终指标：

| 指标 | 数值 |
|---|---:|
| Test accuracy | 0.9370999932 |
| Test loss | 0.1998353451 |
| Parameters | 2,270,794 |
| FLOPs（batch 1） | 599,594,364 |
| Model size | 9.1617 MiB |
| Weight memory | 8.6624 MiB |
| Single mean latency | 2.2894 ms |
| Single median latency | 2.0061 ms |
| Single P95 latency | 3.2166 ms |
| batch 32 mean latency | 11.1140 ms |
| batch 32 P95 latency | 11.5062 ms |
| batch 32 throughput | 2,879.25 images/s |
| process RSS delta | 0.0273 MiB |
| GPU allocator current | 213.51 MiB |
| GPU allocator peak | 616.51 MiB |

运行环境：

- WSL2。
- Linux kernel 字符串：`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- NVIDIA GeForce RTX 4070 Ti。
- 12,282 MiB 总显存。
- NVIDIA driver 610.88。
- Python 3.11.16。
- TensorFlow 2.15.1。
- Keras 2.15.0。
- CUDA/cuDNN 由 TensorFlow 2.15.1 GPU 环境提供。

延迟会因 WSL 调度波动。最终提交以当前 `baseline_metrics.json` 为准，不要在 README 或报告里混用不同次 benchmark 的数字。

能耗：

```text
energy_j_per_inference = null
```

原因已写入 JSON：WSL GPU 进程中没有可靠的逐推理能耗计数器。不要把它改写成估计值，也不要在报告中声称是实测能耗。

## 11. 完整文件和输出清单

### 11.1 主文件

```text
MAIE5532_Assignment3_LuoXiaoyu/
├── requirements.txt
└── part1/
    ├── part1_baseline_model.py
    ├── README.md
    ├── baseline_mobilenetv2.keras
    └── part1_results/
        ├── baseline_metrics.json
        ├── model_summary.txt
        ├── stage1_best.keras
        ├── stage2_best.keras
        ├── training_curves.png
        ├── training_history.json
        ├── training_manifest.json
        ├── training_stage1_log.csv
        └── training_stage2_log.csv
```

文件大小参考：

- 最终模型：9,606,752 bytes。
- Stage 1 checkpoint：9,716,686 bytes。
- Stage 2 checkpoint：27,271,840 bytes。
- 训练曲线 PNG：141,962 bytes。

## 12. 环境与执行方式

### 12.1 重要环境事实

WSL：

```text
Ubuntu
```

系统 Python：

```text
/usr/bin/python3.11
```

系统 Python 中没有 TensorFlow，不要用它运行模型。

实际使用虚拟环境：

```text
/home/mywslubuntu/.venvs/maie5532/bin/python
```

环境中已有：

- TensorFlow 2.15.1。
- tensorflow-model-optimization 0.8.0。
- tf_keras 2.15.1。
- NumPy 1.26.4。
- Matplotlib 3.8.4。
- psutil 7.2.2。

CIFAR-10 已缓存在 WSL 用户目录的 Keras 数据缓存中。MobileNetV2 ImageNet 权重也已在首次训练时下载并缓存。

### 12.2 Codex sandbox 注意

直接从 Windows PowerShell 调 WSL 曾因 sandbox 报 `Wsl/Service/EnumerateDistros/E_ACCESSDENIED`。需要运行 WSL 训练或 GPU 命令时，应使用 `sandbox_permissions="require_escalated"` 并说明用途。

建议命令形式：

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu && /home/mywslubuntu/.venvs/maie5532/bin/python part1/part1_baseline_model.py --benchmark-only"
```

Windows 侧 bundled Python 可用于语法编译检查，但通常没有 TensorFlow，不能做真实模型测试。

### 12.3 常用命令

进入提交目录：

```bash
cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
source ~/.venvs/maie5532/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

从头训练：

```bash
python part1/part1_baseline_model.py
```

只复测已有最终模型：

```bash
python part1/part1_baseline_model.py --benchmark-only
```

使用第二阶段最佳 checkpoint 补齐结果：

```bash
python part1/part1_baseline_model.py --finalize-from-checkpoint
```

只训练不 benchmark：

```bash
python part1/part1_baseline_model.py --no-benchmark
```

## 13. Part 1 已验证和已知问题

已验证：

- 正式的 4+14 epoch 两阶段训练流程实际执行过。
- 两阶段 checkpoint 能保存和加载。
- 最终 `.keras` 能独立加载。
- 最终模型输入 `(None, 224, 224, 3)`。
- 最终模型输出 `(None, 10)`。
- 参数量一致。
- 完整 10,000 张测试集评估过。
- FLOPs profiler 可运行。
- GPU 分配器内存可读取。
- benchmark JSON 可生成。

已知但不阻塞：

- 直接 `keras.models.load_model(final_model)` 可能在 Keras 2.15 下输出 optimizer state warning。权重和推理正常。
- 后续只做推理或迁移时，优先 `load_model(path, compile=False)`，避免无意义的 optimizer 恢复。
- `.keras` 文件可能包含部分训练/优化器状态，因此文件大小不等于纯权重大小。
- 延迟受到 WSL 调度影响，不同运行会出现明显波动。
- TensorFlow profiler 是接近 TF1 的兼容 API，会打印 deprecation warning，但当前能得到 FLOPs。
- WSL 会打印 CUDA NUMA、subprocess permission、TF-TRT 等环境警告，不影响实际 GPU 计算。

已删除：

- Python `__pycache__`。
- 1+1 epoch 冒烟测试的模型与结果。

不要重新引入中间 smoke 结果作为最终交付。

## 14. Part 2 要求和待处理歧义

Part 2 占 30 分。

正文明确列出的部分：

### 14.1 模型架构优化

`create_optimized_models()` 要产生：

- `latency_optimized`
- `memory_optimized`
- `energy_optimized`

延迟优化建议：

- MobileNetV2 `alpha=0.5`。
- 输入降到 160×160 或 128×128。
- 可行时使用 grouped convolution。

内存优化建议：

- structured/unstructured pruning。
- 从 baseline 做 knowledge distillation。
- 系统性地减少各层 filters。

能耗优化建议：

- quantization-aware training。
- early exit。
- 更高效的 activation function。

### 14.2 量化

要求至少包括：

- PTQ INT8。
- QAT INT8。
- mixed precision quantization。
- dynamic range quantization。

代表性校准集应产生 100-500 个覆盖数据分布的样本。

### 14.3 内存管理

要求包括：

- gradient checkpointing。
- model sharding。
- activation compression。
- 根据内存约束选择最优 batch size。

### 14.4 Part 2 交付物

- 所有优化策略的实现。
- 量化模型变体。
- 准确率保持分析。
- 内存优化和实测改进。
- 全部优化变体的性能比较。

### 14.5 必须注意的作业歧义

Part 2 交付物原文写：

```text
Implementation of all four optimization strategies
```

但正文只显式列出三个二级小节：

1. 模型架构优化。
2. 量化。
3. 内存管理。

当前没有老师澄清。

接手 agent 的可执行处理方式：

- 不要静默忽略这个歧义。
- 实现正文所有具体要求：3 类架构变体、4 类量化、4 类内存优化。
- 在 README 或报告中说明原文“four”与三个小节编号不一致。
- 可以把“架构、量化、内存、性能分析/平台调优”解释为四个策略，但必须明确写出这个假设。
- 如果用户能联系老师，建议确认是否有一个未写出的第四类策略。

## 15. Part 3 Track B 要求

Part 3 占 25 分。

必须为 3 种以上平台类型建立综合性能模型。

建议的最小平台集合：

1. native x86 CPU，标记 `[native-x86]`。
2. QEMU ARM Cortex-A，标记 `[QEMU-ARM]`。
3. Renode Cortex-M，运行 INT8 TFLite Micro，标记 `[Renode-CM]`。
4. 可选 WebGPU proxy，标记 `[WebGPU-proxy]`，或者 Vulkan proxy `[Vulkan-proxy]`。

Track B 交付物：

- 3+ 平台类型的综合性能模型。
- 使用多个工具验证，要求至少覆盖 QEMU、Renode、WebGPU 等工具中的多个。
- 跨平台优化效果分析。
- 带文献验证的详细理论分析。

Track B 的约束：

- 所有指标必须标记执行环境。
- 记录版本、命令和 flags，保证可复现。
- 仿真/代理结果只能做相对比较。
- 讨论仿真限制。
- 估计值必须标记为 `"estimated"` 并说明方法。
- 优先使用仿真器中的经验测量，不要只做纯理论。

推荐的主要模型：

- Roofline 模型。
- 基于运算次数的能耗模型。
- 内存访问模型。
- 热降频和功耗约束。
- 跨平台 speedup、memory reduction、energy reduction、accuracy loss 分析。

## 16. Part 4 要求

Part 4 占 20 分。

需要性能比较和硬件感知设计建议，包括：

- performance comparison tables。
- radar charts。
- accuracy 与 efficiency 的 Pareto frontier。
- 不同 batch size 的 scaling。
- CPU SIMD utilization。
- memory bandwidth efficiency。
- cache efficiency。
- thermal characteristics。
- power efficiency。

分析报告长度为 3-4 页，必须覆盖六个主题：

1. Hardware-Software Co-Design Analysis。
2. Platform-Specific Optimization Insights。
3. Energy-Latency-Accuracy Trade-off Analysis。
4. Scalability and Deployment Considerations。
5. Design Methodology Recommendations。
6. Future Hardware Trends Impact。

报告要求包括：

- 硬件约束如何影响模型架构。
- accuracy 与硬件效率的 trade-off。
- 不同优化技术与硬件特性的交互。
- 各平台最有效的优化。
- memory hierarchy 的影响。
- SIMD、tensor core 等专用硬件特性的作用。
- Pareto frontier 图。
- 优化方法适合哪些应用。
- battery-powered edge devices 的影响。
- 跨硬件代际扩展性。
- heterogeneous hardware 部署挑战。
- 资源受限环境的模型更新。
- 系统化设计方法。
- 工具和框架。
- 硬件约束如何进入 ML lifecycle。
- edge AI chips、neuromorphic computing 等未来趋势。

## 17. 全作业最终提交清单

代码：

- `part1_baseline_model.py`：已经完成。
- `part2_optimizations.py`：待完成。
- `part3_modeling.py`：Track B，待完成。
- `performance_profiler.py`：待完成或从现有实现抽取。
- `optimization_framework.py`：待完成。
- `requirements.txt`：已创建。
- `Makefile` 或构建/仿真脚本：待完成。

模型：

- `baseline_mobilenetv2.keras`：已完成。
- `optimized_models/`：待完成。
- 平台模型文件 `.tflite`、`.onnx` 等：待完成。

分析文档：

- `hardware_aware_analysis.pdf`：待完成，3-4 页。
- `performance_benchmarks.xlsx`：待完成。
- `optimization_guidelines.md`：待完成。

演示：

- 根级 `README.md`：待完成。
- `demo_notebook.ipynb`：待完成。
- 平台特定 demo scripts：待完成。
- 3-5 分钟视频：可选，不是必须。

## 18. 目录和命名待确认项

当前按 assignment 2 的方式创建：

```text
MAIE5532_Assignment3_LuoXiaoyu/part1/
```

作业总提交清单写的是文件名，例如 `part1_baseline_model.py`、`baseline_mobilenetv2.keras`，没有明确要求它们必须在压缩包根目录。

最终打包前需要确认：

- 老师是否要求所有脚本和模型平铺到提交根目录。
- 是否要求 zip。
- 学生姓名拼写是否应保持 `LuoXiaoyu`。
- 是否需要排除 checkpoint、缓存和训练中间文件。

当前没有进行任何 `.git` commit 或 zip 打包。

## 19. 接手 agent 的推荐执行顺序

1. 阅读本文件。
2. 检查 `part1/part1_results/baseline_metrics.json` 和最终模型是否存在。
3. 如果只是继续开发，不重新训练 Part 1。
4. 在修改 Part 2 前，检查用户的最新消息是否要求开始 Part 2。
5. 先设计 `part2_optimizations.py` 的输出目录、模型命名和指标 schema。
6. 复用 Part 1 的：
   - 数据划分。
   - 固定 seed。
   - resize 和 Rescaling。
   - test data 不参与训练。
   - latency/accuracy/FLOPs/parameter/metrics JSON 口径。
7. 对 Part 2 明确实现正文所有策略，不要因为“four strategies”歧义漏掉量化或内存管理。
8. 保持中文注释和 docstring。
9. 运行 smoke test，再执行时间较长的训练/转换。
10. 完成 Part 2 后，再进入 Track B 的 Part 3，不要切换到 Track A。

## 20. 不要做的事

- 不要删除或重新生成当前 Part 1 模型，除非用户明确要求。
- 不要把 Track B 改成 Track A。
- 不要直接把当前 `baseline_metrics.json` 中的单样本延迟当作跨平台绝对性能。
- 不要用仿真或代理结果冒充真实硬件测量。
- 不要把估算能耗写成实测能耗。
- 不要在还没完成最终交付时把中文注释转成英文。
- 不要把 test set 用于训练、早停或 checkpoint 选择。
- 不要静默把 MobileNetV2 的 `imagenet` 权重改成 `weights=None`。
- 不要在没有确认的情况下重构或删除用户已有的 assignment 1/2 文件。
- 不要在最终报告中声称训练跑满了 8+25 epoch；实际是 4+14，并采用第二阶段最佳 checkpoint。

## 21. 快速事实表

| 项目 | 当前事实 |
|---|---|
| 路线 | Track B |
| 当前完成阶段 | Part 1 |
| 基线模型 | MobileNetV2，ImageNet 预训练 |
| 数据集 | CIFAR-10 |
| 输入 | 224×224×3 |
| 输出 | 10 类 softmax |
| Test accuracy | 93.71% |
| Accuracy requirement | >85% |
| Parameters | 2,270,794 |
| FLOPs | 599,594,364 |
| Final model MiB | 9.1617 |
| Single mean latency | 2.2894 ms |
| Batch 32 throughput | 2,879.25 images/s |
| GPUs | RTX 4070 Ti |
| TensorFlow | 2.15.1 |
| Keras | 2.15.0 |
| Python | 3.11.16 |
| 注释语言 | 当前中文，最终要转英文 |
| 下一阶段 | Part 2 |

