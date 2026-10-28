# MobileNetV2 CIFAR-10 Multi-Platform Analysis

## Objective and scope

This report evaluates the Part 1 baseline and the Part 2 variants with four execution contexts: native x86 TFLite, simulated Cortex-M3 TFLite Micro, a desktop WebGPU mobile-GPU proxy, and an analytical ARM Cortex-A78 model. The design follows Track B and does not claim physical MCU or smartphone measurements.

| Execution label | Platform class | Evidence |
|---|---|---|
| `[native-x86]` | Desktop CPU | Measured TFLite inference |
| `[QEMU-CM]` | Cortex-M MCU | Measured user-mode QEMU execution |
| `[WebGPU-proxy]` | Mobile GPU proxy | Measured ONNX Runtime Web/WebGPU |
| `[modeled-ARM]` | Cortex-A mobile CPU | Roofline/energy estimate |

## Model profiles

| Model | Accuracy | Parameters | FLOPs | File MiB |
|---|---:|---:|---:|---:|
| baseline | 93.71% | 2,270,794 | 599,594,364 | 26.008 |
| latency_optimized | 89.34% | 719,034 | 98,080,860 | 8.484 |
| memory_optimized | 84.43% | 719,034 | 98,080,860 | 3.239 |
| energy_optimized | 87.46% | 719,364 | 98,084,760 | 8.502 |
| ptq_int8 | 87.84% | 719,034 | 98,080,860 | 0.937 |
| qat_int8 | 84.49% | 719,034 | 98,080,860 | 0.937 |
| mixed_precision | 89.26% | 719,034 | 98,080,860 | 1.369 |
| dynamic_range | 88.61% | 719,034 | 98,080,860 | 0.838 |

The latency model uses `alpha=0.5` and 160x160 input. MobileNetV2 depthwise separable convolutions reduce both computation and parameter count. The INT8 model is 0.937 MiB, compared with 26.008 MiB for the baseline.

## Analytical models

For each platform, latency is the larger of the compute and memory terms after thermal derating, plus a fixed launch overhead:

```text
L_est = max(F / effective_GOPs, B / effective_GBps)
        / thermal_derate + fixed_overhead
```

The working set is `B = weight_bytes + 2 * activation_bytes`; the factor of two approximates overlapping input and output activation buffers. Energy is estimated as

```text
E_est = (F * pJ_per_FLOP + B * pJ_per_byte) / 1e12
        + idle_power_mW * L_est / 1e6
```

These are deliberately simple, uncalibrated Roofline-style models. They do not model per-kernel cache behavior or instruction-level pipelining.

### Platform estimates

| Platform | Model | Latency ms | Memory MiB | Energy mJ |
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

The optimized model has an arithmetic intensity of approximately 18.2 FLOP/byte. All four modeled platforms therefore sit above their Roofline ridge points and reduce latency primarily by lowering FLOPs. The WebGPU model is also strongly affected by its fixed launch overhead.

## Simulation and proxy validation

Native x86 PTQ latency: 0.663 ms.

QEMU Cortex-M PTQ latency: 946.000 ms (clock resolution 10.0 ms, five measured runs).

WebGPU proxy latency: 4.907 ms mean, 5.000 ms median, and 6.100 ms P95 over 100 runs.

| Platform | Model | Measured ms | Predicted ms | Error % |
|---|---|---:|---:|---:|
| native_x86 | ptq_int8 | 0.663 | 1.527 | 130.27% |
| qemu_cm | ptq_int8 | 946.000 | 355.319 | 62.44% |
| webgpu_proxy | latency_optimized | 4.907 | 3.623 | 26.17% |

Raw model errors are large enough that the analytical estimates should be treated as a planning tool, not as replacements for measurement:

- Native x86 is overestimated mainly because the fixed effective throughput is conservative for XNNPACK, AVX2/AVX-512 kernels, cached weights, and pipelined single-image execution.
- Cortex-M is underestimated because the prebuilt portable TensorFlow Lite Micro kernels do not include the same CMSIS-NN/SIMD exploitation assumed by the effective GOPs figure. Pointer chasing, small tensor operations, and interpreter overhead therefore dominate.
- WebGPU is underestimated by about one quarter because browser submission, queue scheduling, synchronization, and desktop driver behavior add cost beyond the simplified 3.5 ms fixed overhead.

### MCU memory check

The PTQ INT8 model contains 982,152 bytes and TFLite Micro reports 707,028 bytes of arena use from a 1,048,576-byte configured arena. Against the modeled Cortex-M7 limits, the model fits in 2 MiB flash, but the arena does not fit in 512 KiB SRAM. QEMU user mode proves functional execution but does not enforce the physical SRAM limit.

## Cross-platform effectiveness

| Platform | Variant | Speedup | Working-set reduction | Energy reduction | Accuracy loss | Fits workspace |
|---|---|---:|---:|---:|---:|---|
| x86_desktop | latency_optimized | 5.61x | 47.55% | 81.97% | 4.37 pp | True |
| x86_desktop | memory_optimized | 5.61x | 47.55% | 81.97% | 9.28 pp | True |
| x86_desktop | energy_optimized | 5.61x | 47.54% | 81.97% | 6.25 pp | True |
| x86_desktop | ptq_int8 | 5.61x | 84.33% | 82.23% | 5.87 pp | True |
| x86_desktop | mixed_precision | 5.61x | 61.56% | 82.07% | 4.45 pp | True |
| arm_cortex_a78 | latency_optimized | 5.73x | 47.55% | 82.00% | 4.37 pp | True |
| arm_cortex_a78 | memory_optimized | 5.73x | 47.55% | 82.00% | 9.28 pp | True |
| arm_cortex_a78 | energy_optimized | 5.73x | 47.54% | 82.00% | 6.25 pp | True |
| arm_cortex_a78 | ptq_int8 | 5.73x | 84.33% | 82.67% | 5.87 pp | True |
| arm_cortex_a78 | mixed_precision | 5.73x | 61.56% | 82.26% | 4.45 pp | True |
| arm_cortex_m7 | latency_optimized | 5.83x | 47.55% | 80.83% | 4.37 pp | False |
| arm_cortex_m7 | memory_optimized | 5.83x | 47.55% | 80.83% | 9.28 pp | False |
| arm_cortex_m7 | energy_optimized | 5.83x | 47.54% | 80.83% | 6.25 pp | False |
| arm_cortex_m7 | ptq_int8 | 5.83x | 84.33% | 83.12% | 5.87 pp | False |
| arm_cortex_m7 | mixed_precision | 5.83x | 61.56% | 81.70% | 4.45 pp | False |
| webgpu_proxy | latency_optimized | 1.17x | 47.55% | 17.49% | 4.37 pp | True |
| webgpu_proxy | memory_optimized | 1.17x | 47.55% | 17.49% | 9.28 pp | True |
| webgpu_proxy | energy_optimized | 1.17x | 47.54% | 17.49% | 6.25 pp | True |
| webgpu_proxy | ptq_int8 | 1.17x | 84.33% | 17.97% | 5.87 pp | True |
| webgpu_proxy | mixed_precision | 1.17x | 61.56% | 17.68% | 4.45 pp | True |

The architecture and INT8 changes reduce FLOPs by about 83.6% and therefore produce a predicted 5.6-5.8x CPU speedup. The measured native PTQ latency of 0.663 ms is consistent with that direction. WebGPU gains only about 1.17x in the analytical model because launch overhead and a much larger compute budget make the reduced graph less dominant.

INT8 reduces the modeled working set by about 84.3%; memory attributes are most valuable on Cortex-M, where flash and SRAM are explicitly limited. The memory-optimized Keras artifact is 3.239 MiB after serialization, but its inference graph still contains dense zero-valued tensors, so compute and bandwidth are not reduced beyond the width change. The energy-optimized early-exit path affects only the 0.81% of samples that pass its confidence threshold; the graph-level analytical model therefore uses the same full-model work for most samples.

## Theoretical validation and interpretation

The Roofline model predicts that reducing arithmetic work should matter until the workload becomes memory- or overhead-bound. MobileNetV2 uses inverted residual blocks and depthwise separable convolution to reduce multiply-accumulates; INT8 quantization replaces float storage and integerizable compute, which is visible in the measured 0.937 MiB model size. These changes reduce both arithmetic work and storage demand.

The measured disagreements also identify the missing effects: cache reuse, SIMD/kernel selection, framework dispatch, and accelerator scheduling. A production model would calibrate effective throughput per operation type and use empirical microbenchmarks. That calibration is intentionally omitted because the raw model is the object being validated.

## Limitations

- QEMU-CM is user-mode simulation; it does not enforce SRAM, flash, cache, power, clock-frequency, or thermal limits.
- The WebGPU result uses a desktop GPU and synthetic input as a relative proxy for a mobile GPU.
- Energy values are analytical estimates derived from operation and byte counts; no physical power meter was used.
- Floating-point equal-FLOP assumptions do not capture Tensor Core or NPU execution.
- Only one INT8 model was executed on Cortex-M. The other model rows are analytical predictions.

Qualified execution labels:

- `[native-x86]`: TFLite CPU measurement.
- `[QEMU-CM]`: qemu-arm Cortex-M3 with TFLite Micro.
- `[WebGPU-proxy]`: ONNX Runtime Web on desktop WebGPU.
- `[modeled-ARM]`: analytical ARM Cortex-A estimate.

QEMU and WebGPU results are relative simulation evidence, not physical MCU or smartphone measurements.

## References

1. S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, 2009.
2. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.
3. B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018.
4. R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems," *Proceedings of MLSys*, 2021.
