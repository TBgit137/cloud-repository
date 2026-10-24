# 第 3 部分：多平台分析

## 目标和范围

本部分选择 Track B。分析覆盖四种执行环境：本地 x86 TFLite、QEMU 模拟 Cortex-M3、桌面 WebGPU 移动 GPU 代理，以及 Cortex-A78 解析模型。QEMU 和 WebGPU 结果属于仿真或代理证据，不是真机和智能手机测量。

| 标签 | 平台类型 | 证据 |
|---|---|---|
| `[native-x86]` | 桌面 CPU | TFLite 实测 |
| `[QEMU-CM]` | Cortex-M MCU | QEMU 用户模式实测 |
| `[WebGPU-proxy]` | 移动 GPU 代理 | ONNX Runtime Web/WebGPU 实测 |
| `[modeled-ARM]` | Cortex-A 移动 CPU | 仅解析估算 |

## 模型概况

| 模型 | 准确率 | 参数量 | FLOPs | 文件 MiB |
|---|---:|---:|---:|---:|
| baseline | 93.71% | 2,270,794 | 599,594,364 | 26.008 |
| latency_optimized | 89.34% | 719,034 | 98,080,860 | 8.484 |
| memory_optimized | 84.43% | 719,034 | 98,080,860 | 3.239 |
| energy_optimized | 87.46% | 719,364 | 98,084,760 | 8.502 |
| ptq_int8 | 87.84% | 719,034 | 98,080,860 | 0.937 |
| qat_int8 | 84.49% | 719,034 | 98,080,860 | 0.937 |
| mixed_precision | 89.26% | 719,034 | 98,080,860 | 1.369 |
| dynamic_range | 88.61% | 719,034 | 98,080,860 | 0.838 |

## 性能模型

延迟使用 Roofline 思路，取计算时间和访存时间的较大值，再除以热降频系数并加上固定调用开销：

```text
L_est = max(F / 有效GOPs, B / 有效GBps)
        / 热降频系数 + 固定开销
```

工作集近似为 `B = 权重字节数 + 2 × 峰值激活字节数`。能量估算为：

```text
E_est = (F × 每FLOP能耗 + B × 每字节能耗) / 1e12
        + 空闲功率 × 估算延迟
```

这是未经实测校准的简化模型，没有逐层模拟 cache、流水线或实际内核调度。

### 各平台估算

| 平台 | 模型 | 延迟 ms | 内存 MiB | 能量 mJ |
|---|---|---:|---:|---:|
| x86_desktop | baseline | 8.565 | 9.811 | 132.404 |
| x86_desktop | latency_optimized | 1.527 | 5.146 | 23.875 |
| x86_desktop | memory_optimized | 1.527 | 5.146 | 23.875 |
| x86_desktop | energy_optimized | 1.527 | 5.147 | 23.876 |
| x86_desktop | ptq_int8 | 1.527 | 1.537 | 23.534 |
| x86_desktop | qat_int8 | 1.527 | 1.537 | 23.534 |
| x86_desktop | mixed_precision | 1.527 | 3.772 | 23.745 |
| x86_desktop | dynamic_range | 1.527 | 3.241 | 23.695 |
| arm_cortex_a78 | baseline | 75.949 | 9.811 | 101.985 |
| arm_cortex_a78 | latency_optimized | 13.260 | 5.146 | 18.355 |
| arm_cortex_a78 | memory_optimized | 13.260 | 5.146 | 18.355 |
| arm_cortex_a78 | energy_optimized | 13.261 | 5.147 | 18.355 |
| arm_cortex_a78 | ptq_int8 | 13.260 | 1.537 | 17.674 |
| arm_cortex_a78 | qat_int8 | 13.260 | 1.537 | 17.674 |
| arm_cortex_a78 | mixed_precision | 13.260 | 3.772 | 18.095 |
| arm_cortex_a78 | dynamic_range | 13.260 | 3.241 | 17.995 |
| arm_cortex_m7 | baseline | 2069.895 | 9.811 | 148.729 |
| arm_cortex_m7 | latency_optimized | 355.319 | 5.146 | 28.507 |
| arm_cortex_m7 | memory_optimized | 355.319 | 5.146 | 28.507 |
| arm_cortex_m7 | energy_optimized | 355.333 | 5.147 | 28.509 |
| arm_cortex_m7 | ptq_int8 | 355.319 | 1.537 | 25.102 |
| arm_cortex_m7 | qat_int8 | 355.319 | 1.537 | 25.102 |
| arm_cortex_m7 | mixed_precision | 355.319 | 3.772 | 27.210 |
| arm_cortex_m7 | dynamic_range | 355.319 | 3.241 | 26.710 |
| webgpu_proxy | baseline | 4.249 | 9.811 | 35.658 |
| webgpu_proxy | latency_optimized | 3.623 | 5.146 | 29.420 |
| webgpu_proxy | memory_optimized | 3.623 | 5.146 | 29.420 |
| webgpu_proxy | energy_optimized | 3.623 | 5.147 | 29.420 |
| webgpu_proxy | ptq_int8 | 3.623 | 1.537 | 29.250 |
| webgpu_proxy | qat_int8 | 3.623 | 1.537 | 29.250 |
| webgpu_proxy | mixed_precision | 3.623 | 3.772 | 29.355 |
| webgpu_proxy | dynamic_range | 3.623 | 3.241 | 29.330 |

## 仿真和代理验证

本地 x86 PTQ INT8 延迟为 0.663 ms。

QEMU Cortex-M PTQ INT8 延迟为 946.000 ms。固件时钟精度为 10.0 ms，共测量 5 次。

WebGPU 代理延迟为均值 4.907 ms、中位数 5.000 ms、P95 6.100 ms，共测量 100 次。

| 平台 | 模型 | 实测 ms | 预测 ms | 相对误差 |
|---|---|---:|---:|---:|
| native_x86 | ptq_int8 | 0.663 | 1.527 | 130.27% |
| qemu_cm | ptq_int8 | 946.000 | 355.319 | 62.44% |
| webgpu_proxy | latency_optimized | 4.907 | 3.623 | 26.17% |

误差来源如下：

- x86 预测偏慢，因为有效吞吐参数没有充分计入 XNNPACK、AVX2/AVX-512、权重缓存复用和流水线执行。
- Cortex-M 预测偏快，因为当前 TFLite Micro 静态库主要使用可移植内核，没有采用解析模型假定的 CMSIS-NN/SIMD 加速程度。
- WebGPU 预测偏快，因为浏览器提交、队列调度、同步和桌面驱动带来额外开销。

### MCU 内存检查

PTQ INT8 模型为 982,152 bytes，TFLite Micro 实际使用 707,028 bytes arena；配置的 arena 上限为 1,048,576 bytes。相对于示例 Cortex-M7 的 2 MiB flash 和 512 KiB SRAM：

- 模型可以放入 flash。
- arena 不能放入 SRAM。

QEMU 用户模式只能证明程序能够执行，不会真正执行 512 KiB SRAM 上限检查。

## 跨平台优化效果

| 平台 | 变体 | 加速 | 工作集下降 | 能量下降 | 准确率损失 | 满足内存约束 |
|---|---|---:|---:|---:|---:|---|
| x86_desktop | latency_optimized | 5.61x | 47.55% | 81.97% | 4.37 个百分点 | True |
| x86_desktop | memory_optimized | 5.61x | 47.55% | 81.97% | 9.28 个百分点 | True |
| x86_desktop | energy_optimized | 5.61x | 47.54% | 81.97% | 6.25 个百分点 | True |
| x86_desktop | ptq_int8 | 5.61x | 84.33% | 82.23% | 5.87 个百分点 | True |
| x86_desktop | mixed_precision | 5.61x | 61.56% | 82.07% | 4.45 个百分点 | True |
| arm_cortex_a78 | latency_optimized | 5.73x | 47.55% | 82.00% | 4.37 个百分点 | True |
| arm_cortex_a78 | memory_optimized | 5.73x | 47.55% | 82.00% | 9.28 个百分点 | True |
| arm_cortex_a78 | energy_optimized | 5.73x | 47.54% | 82.00% | 6.25 个百分点 | True |
| arm_cortex_a78 | ptq_int8 | 5.73x | 84.33% | 82.67% | 5.87 个百分点 | True |
| arm_cortex_a78 | mixed_precision | 5.73x | 61.56% | 82.26% | 4.45 个百分点 | True |
| arm_cortex_m7 | latency_optimized | 5.83x | 47.55% | 80.83% | 4.37 个百分点 | False |
| arm_cortex_m7 | memory_optimized | 5.83x | 47.55% | 80.83% | 9.28 个百分点 | False |
| arm_cortex_m7 | energy_optimized | 5.83x | 47.54% | 80.83% | 6.25 个百分点 | False |
| arm_cortex_m7 | ptq_int8 | 5.83x | 84.33% | 83.12% | 5.87 个百分点 | False |
| arm_cortex_m7 | mixed_precision | 5.83x | 61.56% | 81.70% | 4.45 个百分点 | False |
| webgpu_proxy | latency_optimized | 1.17x | 47.55% | 17.49% | 4.37 个百分点 | True |
| webgpu_proxy | memory_optimized | 1.17x | 47.55% | 17.49% | 9.28 个百分点 | True |
| webgpu_proxy | energy_optimized | 1.17x | 47.54% | 17.49% | 6.25 个百分点 | True |
| webgpu_proxy | ptq_int8 | 1.17x | 84.33% | 17.97% | 5.87 个百分点 | True |
| webgpu_proxy | mixed_precision | 1.17x | 61.56% | 17.68% | 4.45 个百分点 | True |

## 结论和局限

- 输入从 224 降到 160、宽度乘数降到 0.5 后，FLOPs 从约 599.6M 降到约 98.1M，因此 CPU 解析模型给出约 5.6 到 5.8 倍加速。
- INT8 将工作集估算降低约 84.3%，是将模型压入边缘设备存储预算的主要手段。
- WebGPU 的固定调用开销显著，模型计算量减少并未带来同等比例的端到端加速。
- 能量值来自运算量和访存量的解析公式，不是功率计实测。
- QEMU 不模拟 cache、总线、功耗、温度和物理 SRAM 限制。

课程第 6 周对延迟、内存、能量和量化变体的划分，以及第 7 周关于数据移动和跨平台优化的讨论，与上述结果方向一致。Roofline、MobileNetV2、整数推断量化和 TFLite Micro 的原始文献见英文报告的 References。

## Track B 交付覆盖

| 要求 | 对应产物 |
|---|---|
| 3 种以上平台 | x86、Cortex-M、Cortex-A78、GPU 代理 |
| 多工具验证 | TFLite、TFLite Micro/QEMU、TensorFlow profiler、ONNX Runtime Web/WebGPU |
| 跨平台优化分析 | `cross_platform_analysis.json` 和本文表格 |
| 理论及文献验证 | Roofline/能量/内存模型、实测对照和参考文献 |
| 可复现说明 | `README.md`、`cortex_m/Makefile` |
