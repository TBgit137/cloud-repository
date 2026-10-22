# 硬件感知设计作业

**截止时间：** 第 8 周

## 目标

设计并实现一个硬件感知的机器学习应用，在面向不同部署目标时综合考虑能耗、延迟和内存约束。你需要针对多种硬件平台优化一个计算机视觉模型，实施硬件特定优化，并通过真实硬件部署或完整的仿真与建模来分析性能权衡。

## 背景

本作业建立在第 6-8 周讨论的硬件加速原理之上。你将亲身体验硬件特性如何影响机器学习系统设计，从算法选择、内存管理到能耗优化。作业重点讨论在边缘部署 AI 并同时保持可接受性能所面临的实际挑战。

## 硬件要求说明

**本作业可以完全通过仿真工具和性能建模完成。有专用硬件的同学可以使用该硬件，但获得满分并不要求使用专用硬件。**

你可以选择：

- **路线 A：真实硬件部署** - 在实际硬件平台上部署，并获取真实测量结果。
- **路线 B：仿真与建模** - 使用仿真工具和理论分析，对平台进行全面评估。

两条路线都可以获得满分，并达到全部学习目标。

## 第 1 部分：基线模型实现（25 分）

实现一个基于 MobileNetV2 的图像分类模型，作为后续硬件感知优化的基线。

### 实现要求

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import psutil
import os


def create_baseline_model(input_shape=(224, 224, 3), num_classes=10):
    """
    创建一个用于图像分类的 MobileNetV2 基础模型。

    参数：
        input_shape：输入图像尺寸
        num_classes：分类类别数

    返回：
        tf.keras.Model：已编译、可直接训练的模型
    """
    # TODO：实现 MobileNetV2 架构
    # 基础模型：MobileNetV2（在 ImageNet 上预训练，初始阶段冻结）
    # 为你的数据集添加自定义分类头
    # 使用全局平均池化和 dropout 进行正则化
    pass


def load_and_preprocess_data():
    """
    加载 CIFAR-10 数据集，并为 MobileNetV2 进行预处理。

    返回：
        tuple：(x_train, y_train, x_test, y_test)
    """
    # TODO：加载 CIFAR-10 数据集
    # 将图像缩放到 224x224，以适配 MobileNetV2
    # 归一化像素值，并应用数据增强
    # 使用 tf.data.Dataset 高效加载数据
    pass


def benchmark_baseline_model(model, test_data, batch_size=32):
    """
    对基线模型的延迟和内存使用情况进行基准测试。

    参数：
        model：已训练的 Keras 模型
        test_data：测试数据集
        batch_size：推理时使用的批大小

    返回：
        dict：性能指标
    """
    # TODO：测量单样本和批量推理延迟
    # TODO：监控推理过程中的内存使用
    # TODO：计算 FLOPs 和模型参数量
    # TODO：在平台允许时测量能耗

    metrics = {
        'single_inference_time': 0.0,
        'batch_inference_time': 0.0,
        'memory_usage_mb': 0.0,
        'model_size_mb': 0.0,
        'accuracy': 0.0,
        'flops': 0,
        'parameters': 0
    }
    return metrics


if __name__ == "__main__":
    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # 创建并训练模型
    model = create_baseline_model()

    # 使用微调策略进行训练
    # 1. 仅训练分类头（冻结基础模型）
    # 2. 解冻基础模型，并使用较低学习率进行微调

    # 保存训练好的模型
    model.save('baseline_mobilenetv2.keras')

    # 进行基准测试
    metrics = benchmark_baseline_model(model, x_test)
    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
```

### 第 1 部分交付物

- 完整的 MobileNetV2 基线模型实现。
- 展示迁移学习过程的训练日志。
- 达到的测试准确率（目标：在 CIFAR-10 上高于 85%）。
- 基线性能指标，包括延迟、内存和模型大小。

## 第 2 部分：硬件感知优化（30 分）

实现多种针对不同硬件约束的优化技术。

### 2.1 模型架构优化

```python
def create_optimized_models():
    """
    创建基线模型针对不同硬件条件优化的变体。

    返回：
        dict：面向不同约束优化后的模型
    """
    models = {}

    # TODO：实现延迟优化模型
    # - 降低 MobileNetV2 的深度乘子
    # - 使用逐深度可分离卷积
    # - 针对推理速度进行优化
    models['latency_optimized'] = create_latency_optimized_model()

    # TODO：实现内存优化模型
    # - 降低模型宽度（通道数）
    # - 使用模型剪枝技术
    # - 优化内存占用
    models['memory_optimized'] = create_memory_optimized_model()

    # TODO：实现能耗优化模型
    # - 使用量化感知训练
    # - 降低计算强度
    # - 优化能效
    models['energy_optimized'] = create_energy_optimized_model()

    return models


def create_latency_optimized_model():
    """创建针对推理延迟优化的模型。"""
    # TODO：使用 alpha=0.5 的 MobileNetV2（深度乘子）
    # TODO：将输入分辨率降低到 160x160 或 128x128
    # TODO：在适用时使用分组卷积
    pass


def create_memory_optimized_model():
    """创建针对内存占用优化的模型。"""
    # TODO：实现模型剪枝（结构化和非结构化）
    # TODO：使用基线模型进行知识蒸馏
    # TODO：系统地减少每一层的滤波器数量
    pass


def create_energy_optimized_model():
    """创建针对能耗优化的模型。"""
    # TODO：实现量化感知训练
    # TODO：使用提前退出机制
    # TODO：优化激活函数以提高效率
    pass
```

### 2.2 量化实现

```python
def apply_quantization_optimizations(model, x_train_sample):
    """
    对模型应用不同的量化技术。

    参数：
        model：已训练的 Keras 模型
        x_train_sample：用于校准的代表性数据集

    返回：
        dict：不同量化模型变体
    """
    quantized_models = {}

    # TODO：训练后量化（PTQ）
    quantized_models['ptq_int8'] = post_training_quantization(model, x_train_sample)

    # TODO：量化感知训练（QAT）
    quantized_models['qat_int8'] = quantization_aware_training(model, x_train_sample)

    # TODO：混合精度量化
    quantized_models['mixed_precision'] = mixed_precision_quantization(model)

    # TODO：动态范围量化
    quantized_models['dynamic_range'] = dynamic_range_quantization(model)

    return quantized_models


def representative_dataset_generator(x_train_sample):
    """生成用于量化校准的数据。"""
    # TODO：实现用于校准的代表性数据集
    # 应产出 100-500 个覆盖数据分布的样本
    pass
```

### 2.3 内存管理优化

```python
def implement_memory_optimizations(model):
    """
    实施内存感知优化。

    参数：
        model：待优化的基础模型

    返回：
        dict：内存优化结果
    """
    optimizations = {}

    # TODO：实现梯度检查点
    optimizations['gradient_checkpointing'] = implement_gradient_checkpointing(model)

    # TODO：为大模型实现模型分片
    optimizations['model_sharding'] = implement_model_sharding(model)

    # TODO：实现激活值压缩
    optimizations['activation_compression'] = implement_activation_compression(model)

    # TODO：针对内存约束优化批大小
    optimizations['optimal_batch_size'] = find_optimal_batch_size(model)

    return optimizations
```

### 第 2 部分交付物

- 实现全部四种优化策略。
- 量化模型变体及准确率保持情况分析。
- 内存优化技术及实测改进结果。
- 所有优化变体之间的性能比较。

## 第 3 部分：多平台分析（25 分）

根据你可使用的硬件，在**路线 A**或**路线 B**中选择一条。

### 路线 A：真实硬件部署

将优化后的模型部署到不同硬件平台，并测量真实性能。

#### 3.1 平台特定优化

```python
class HardwareOptimizer:
    """面向不同平台的硬件感知模型优化器。"""

    def __init__(self, target_platform):
        self.platform = target_platform
        self.optimization_config = self._get_platform_config()

    def _get_platform_config(self):
        """获取目标平台的优化配置。"""
        configs = {
            'cpu_x86': {
                'optimization_level': 'O3',
                'use_avx': True,
                'thread_count': 4,
                'memory_constraint_mb': 1024
            },
            'arm_cortex_a': {
                'use_neon': True,
                'fp16_acceleration': True,
                'memory_constraint_mb': 512,
                'power_budget_mw': 2000
            },
            'arm_cortex_m': {
                'quantization': 'int8',
                'memory_constraint_kb': 256,
                'power_budget_mw': 50,
                'use_cmsis_nn': True
            },
            'gpu_mobile': {
                'use_gpu_delegate': True,
                'fp16_inference': True,
                'memory_constraint_mb': 2048,
                'thermal_throttling': True
            }
        }
        return configs.get(self.platform, {})

    def optimize_for_platform(self, model):
        """应用平台特定的优化。"""
        # TODO：实现平台特定的优化流程
        # CPU：使用 Intel OpenVINO 或 ARM Compute Library 优化
        # ARM Cortex-A：使用 NEON SIMD 指令和 FP16
        # ARM Cortex-M：使用 CMSIS-NN 和激进量化
        # 移动 GPU：使用 GPU 委托（delegate）和纹理内存优化
        pass


# TODO：实现平台特定的部署
def deploy_to_tflite_micro(model, target_mcu='cortex_m4'):
    """将模型部署到微控制器使用的 TensorFlow Lite Micro。"""
    pass


def deploy_to_mobile_gpu(model, target_gpu='adreno_640'):
    """使用 GPU 加速部署模型。"""
    pass


def deploy_to_edge_tpu(model):
    """将模型部署到 Google Coral Edge TPU。"""
    pass
```

#### 路线 A 交付物

- 至少在 3 种不同类型的平台上完成可运行的部署。
- 平台特定的优化实现。
- 所有平台上的真实性能测量结果。
- 展示持续性能的压力测试结果。

### 路线 B：仿真与性能建模

开发完整的性能模型，并使用仿真工具进行验证。

#### 3.1 平台性能建模

```python
class PlatformPerformanceModel:
    """对不同硬件平台的性能特征进行建模。"""

    def __init__(self, platform_specs):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_performance_models(self):
        """为每个平台构建解析性能模型。"""
        models = {}

        # TODO：为每个平台实现 Roofline 模型
        # 考虑内存带宽、计算吞吐量和缓存层级
        models['roofline'] = self._build_roofline_model()

        # TODO：实现基于运算次数的能耗模型
        models['energy'] = self._build_energy_model()

        # TODO：实现内存访问模型
        models['memory'] = self._build_memory_model()

        return models

    def estimate_performance(self, model_graph, platform_type):
        """估计模型在某个平台上的性能指标。"""
        # TODO：分析模型图的 FLOPs 和内存访问模式
        # TODO：应用平台特定的性能模型
        # TODO：考虑热降频和功耗约束
        pass


def validate_with_simulation(models, platform_configs):
    """使用仿真工具验证性能估计结果。"""

    validation_results = {}

    # TODO：使用 QEMU 进行 ARM 仿真
    validation_results['qemu_arm'] = simulate_arm_performance(models)

    # TODO：使用 Renode 进行 Cortex-M 仿真
    validation_results['renode_cm'] = simulate_cortex_m_performance(models)

    # TODO：使用 WebGPU 作为移动 GPU 的代理
    validation_results['webgpu_proxy'] = simulate_mobile_gpu_performance(models)

    # TODO：将仿真结果与解析模型交叉验证
    validation_results['model_accuracy'] = cross_validate_models(validation_results)

    return validation_results


def simulate_arm_performance(models):
    """使用 QEMU 仿真 ARM Cortex-A 性能。"""
    # TODO：搭建 QEMU ARM 环境
    # TODO：部署 TFLite 模型并测量性能
    # TODO：提取详细的性能指标
    pass


def simulate_cortex_m_performance(models):
    """使用 Renode 仿真 Cortex-M 性能。"""
    # TODO：搭建 Renode 仿真环境
    # TODO：部署 TFLite Micro 模型
    # TODO：测量周期数和内存使用量
    pass


def simulate_mobile_gpu_performance(models):
    """使用 WebGPU 代理仿真移动 GPU 性能。"""
    # TODO：将模型转换为 WebGPU 兼容格式
    # TODO：在浏览器环境中测量性能
    # TODO：根据移动 GPU 规格对结果进行缩放
    pass
```

#### 3.2 跨平台分析框架

```python
class CrossPlatformAnalyzer:
    """分析优化在不同平台上的有效性。"""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()

    def _load_platform_specs(self):
        """加载目标平台的详细规格。"""
        return {
            'x86_desktop': {
                'core_count': 8,
                'frequency_ghz': 3.5,
                'simd_width': 256,  # AVX2 指令集
                'l1_cache_kb': 32,
                'l2_cache_kb': 256,
                'l3_cache_mb': 8,
                'memory_bandwidth_gbps': 50,
                'tdp_watts': 65
            },
            'arm_cortex_a78': {
                'core_count': 4,
                'frequency_ghz': 2.4,
                'simd_width': 128,  # NEON 指令集
                'l1_cache_kb': 64,
                'l2_cache_kb': 512,
                'memory_bandwidth_gbps': 15,
                'tdp_watts': 5
            },
            'arm_cortex_m7': {
                'frequency_mhz': 400,
                'sram_kb': 512,
                'flash_mb': 2,
                'power_budget_mw': 100,
                'has_fpu': True,
                'dsp_extensions': True
            }
        }

    def analyze_optimization_effectiveness(self, baseline_model, optimized_models):
        """分析不同优化在各平台上的表现。"""

        results = {}

        for platform in self.platform_characteristics:
            platform_results = {}

            # TODO：对基线模型在平台上的性能进行建模
            baseline_perf = self.model_performance(baseline_model, platform)

            # TODO：对优化变体的性能进行建模
            for opt_name, opt_model in optimized_models.items():
                opt_perf = self.model_performance(opt_model, platform)
                platform_results[opt_name] = {
                    'speedup': baseline_perf['latency'] / opt_perf['latency'],
                    'memory_reduction': 1 - (opt_perf['memory'] / baseline_perf['memory']),
                    'energy_reduction': 1 - (opt_perf['energy'] / baseline_perf['energy']),
                    'accuracy_loss': baseline_perf['accuracy'] - opt_perf['accuracy']
                }

            results[platform] = platform_results

        return results

    def model_performance(self, model, platform):
        """对模型在某个平台上的预期性能进行建模。"""
        # TODO：实现详细的性能建模
        # 考虑指令级分析、内存访问模式
        # 缓存行为和热约束
        pass
```

#### 路线 B 交付物

- 针对 3 种以上平台类型的完整性能模型。
- 使用多种工具（QEMU、Renode、WebGPU）进行的仿真验证。
- 跨平台优化有效性分析。
- 结合文献验证的详细理论分析。

## 第 4 部分：综合分析与设计建议（20 分）

### 4.1 性能分析

创建详细的性能比较，并提出硬件感知设计建议。

```python
def generate_performance_report():
    """生成完整的性能分析报告。"""

    # TODO：创建性能比较表
    performance_data = {
        'models': ['baseline', 'latency_opt', 'memory_opt', 'energy_opt'],
        'platforms': ['cpu_x86', 'arm_cortex_a', 'arm_cortex_m', 'mobile_gpu'],
        'metrics': ['latency_ms', 'memory_mb', 'energy_mj', 'accuracy', 'throughput_fps']
    }

    # TODO：生成用于多维权衡分析的雷达图
    # TODO：创建准确率与效率的帕累托前沿分析
    # TODO：分析不同批大小下的性能缩放情况

    return performance_data


def analyze_hardware_utilization():
    """分析每个模型对不同硬件特性的利用程度。"""

    utilization_analysis = {
        'cpu_simd_utilization': {},  # SIMD 指令使用情况
        'memory_bandwidth_efficiency': {},  # 内存访问模式
        'cache_efficiency': {},  # L1/L2 缓存命中率
        'thermal_characteristics': {},  # 温度分布特征
        'power_efficiency': {}  # 每次运算的能耗
    }

    return utilization_analysis
```

### 4.2 设计方法

```python
class HardwareAwareDesignMethodology:
    """硬件感知机器学习系统设计框架。"""

    def __init__(self):
        self.design_principles = [
            'co_design_hardware_software',
            'early_constraint_specification',
            'iterative_optimization',
            'multi_objective_optimization',
            'platform_specific_tuning'
        ]

    def constraint_specification_framework(self, application_requirements):
        """用于在设计早期明确硬件约束的框架。"""
        # TODO：定义约束描述方法
        # 功耗预算、内存限制、延迟要求和准确率目标
        pass

    def optimization_priority_framework(self, constraints):
        """根据约束确定优化优先级。"""
        # TODO：为优化策略选择创建决策树
        pass

    def design_space_exploration(self, base_model, constraints):
        """对设计方案进行系统化探索。"""
        # TODO：实现自动化设计空间探索
        # 架构变体、量化级别和优化技术
        pass
```

## 必需的分析报告（3-4 页）

### 1. 软硬件协同设计分析

- 硬件约束如何影响你的模型架构决策？
- 你观察到准确率与硬件效率之间存在哪些权衡？
- 不同优化技术与硬件特性如何相互影响？

### 2. 平台特定优化洞察

- 哪些优化对各类平台最有效？
- 内存层级差异如何影响优化策略？
- 专用硬件特性（SIMD、Tensor Core）在性能中发挥了什么作用？

### 3. 能耗、延迟与准确率的权衡分析

- 创建帕累托前沿图，展示各项指标之间的权衡。
- 分析哪些应用会从每种优化方法中受益。
- 讨论这些结论对电池供电边缘设备的影响。

### 4. 可扩展性与部署考虑

- 你的优化在不同代际硬件之间如何扩展？
- 在异构硬件上部署时会出现哪些挑战？
- 你将如何处理资源受限环境中的模型更新？

### 5. 设计方法建议

- 提出一种面向硬件感知机器学习系统设计的系统化方法。
- 哪些工具和框架可以改进硬件感知设计流程？
- 应如何将硬件约束纳入机器学习开发生命周期？

### 6. 未来硬件趋势的影响

- 新兴硬件趋势（边缘 AI 芯片、类脑计算）可能如何影响你的设计决策？
- 你预见到哪些新的优化机会？

## 提交要求

### 代码交付物

- `part1_baseline_model.py` - MobileNetV2 基线模型实现。
- `part2_optimizations.py` - 硬件感知优化。
- `part3_deployment.py`（路线 A）或 `part3_modeling.py`（路线 B） - 平台分析代码。
- `performance_profiler.py` - 性能测量工具。
- `optimization_framework.py` - 硬件感知设计框架。
- `requirements.txt` - Python 依赖。
- 用于部署或仿真的 `Makefile` 或构建脚本。

### 模型文件

- `baseline_mobilenetv2.keras` - 已训练的基线模型。
- `optimized_models/` - 包含所有优化模型变体的目录。
- 平台特定模型文件（`.tflite`、`.onnx` 等）。

### 分析文档

- `hardware_aware_analysis.pdf` - 综合分析报告（3-4 页）。
- `performance_benchmarks.xlsx` - 详细的性能测量结果。
- `optimization_guidelines.md` - 设计方法文档。

### 演示材料

- `README.md` - 完整的安装与运行说明。
- `demo_notebook.ipynb` - 展示关键结果的 Jupyter notebook。
- 平台特定的演示脚本。
- 视频演示（可选，3-5 分钟）。

## 硬件获取与仿真资源

### 路线 A：推荐硬件平台

- **CPU：** Intel/AMD x86、ARM Cortex-A 系列。
- **移动 GPU：** Qualcomm Adreno、ARM Mali、Apple GPU。
- **边缘 AI：** Google Coral Dev Board、Intel Neural Compute Stick。
- **MCU：** Arduino Nano 33 BLE、STM32 Discovery 开发板。

### 路线 B：仿真替代方案

如果没有实体硬件，可以使用：

- **ARM Cortex-A（Linux 级）：** 使用 QEMU（系统模式或用户模式），在 ARM 根文件系统中运行 ARM 二进制文件（例如 TFLite/ONNX Runtime）。将结果标记为 `[QEMU-ARM]`。
- **ARM Cortex-M（MCU 级）：** 使用 Renode（推荐）或 QEMU Cortex-M，运行采用 INT8 模型的 TensorFlow Lite Micro 固件；从仿真器日志中报告周期数或延迟。标记为 `[Renode-CM]` 或 `[QEMU-CM]`。
- **移动 GPU 代理：** 在桌面 Chromium 浏览器中使用带 WebGPU 的 ONNX Runtime Web，或在桌面端使用 ncnn（Vulkan）作为移动 GPU 的代理。标记为 `[WebGPU-proxy]` 或 `[Vulkan-proxy]`。
- **x86 CPU：** 使用 TFLite 或 ONNX Runtime 在桌面或笔记本电脑上原生运行，作为基线比较。标记为 `[native-x86]`。

### 路线 B 注意事项

- 明确标记所有报告指标的执行环境（例如 `[QEMU-ARM]`）。
- 记录所使用的版本、命令和参数，以确保结果可复现。
- 将仿真或代理结果视为相对比较，并在报告中讨论其局限性。
- 明确将估算得到的数值标记为 "estimated"，并说明估算方法。
- 优先采用经验测量（即使是在仿真器中），而非纯理论计算。

### 性能估算工具（两条路线均适用）

- **TensorFlow Model Optimization Toolkit（TF-MOT）：** 分析剪枝和量化的效果。
- **TFLite `benchmark_model` 和 `onnxruntime_perf_test`：** 执行可重复的延迟测试。
- **FLOPs 与参数量估算器：** `tensorflow.python.profiler`、`keras-flops` 或 Netron。
- **功耗与能耗工具：** Linux 中的 `powertop` 或 `turbostat`，用于估算封装功耗。
- **公开基准结果：** 参考厂商或社区结果，作为比较背景。
