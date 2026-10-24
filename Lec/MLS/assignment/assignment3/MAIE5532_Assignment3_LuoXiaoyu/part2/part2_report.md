# Part 2 Hardware-Aware Optimization Report

## 1. Overview

This part builds on the 93.71% CIFAR-10 MobileNetV2 baseline from Part 1.
Four primary optimization variants are evaluated:

1. Latency-optimized model.
2. Memory-optimized model.
3. Energy-efficient model with an early-exit branch.
4. Quantized models.

The first three are Keras models. Quantized variants are TFLite models.
All accuracy measurements use the complete 10,000-image CIFAR-10 test set.
The test set was not used for training or calibration.

## 2. Implementation Summary

### 2.1 Latency Optimization

- MobileNetV2 width multiplier: `alpha=0.5`.
- Input resolution: `160 x 160`.
- Parameters: 719,034.
- The model retains MobileNetV2's depthwise-separable convolution structure.

### 2.2 Memory Optimization

- MobileNetV2 width multiplier: `alpha=0.5`.
- Input resolution: `160 x 160`.
- Knowledge distillation from the Part 1 baseline teacher.
- Masked magnitude pruning of convolution and Dense kernels.
- Target and measured kernel sparsity: approximately 30%.
- Two masked recovery epochs preserve zero weights during fine-tuning.

### 2.3 Energy Optimization

- MobileNetV2 width multiplier: `alpha=0.5`.
- Input resolution: `160 x 160`.
- An early-exit classifier is attached to `block_6_project`.
- Samples with confidence at least 0.7 use the early classifier.
- Low-confidence samples continue through the full network.
- No per-inference energy counter was available, so energy efficiency is
  discussed through latency, early-exit coverage, and reduced computation
  rather than measured joules.

### 2.4 Quantization

- PTQ INT8 uses 200 training images for calibration.
- QAT applies an 8-bit weight-grid constraint during short training.
- Mixed precision exports an FP16 TFLite model.
- Dynamic-range quantization reduces weight precision without calibration.

## 3. Model Comparison

The baseline timing uses the Keras GPU model. Architecture-model timings also
use Keras. Quantized-model timings use a single-threaded TFLite CPU
interpreter. Therefore, quantized TFLite latency is not a direct same-backend
hardware speedup over the Keras GPU baseline.

| Model | Backend | Accuracy | Drop vs baseline | Parameters | Size (MiB) | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Keras GPU | 93.71% | 0.00 pp | 2,270,794 | 26.008 | 2.497 |
| Latency | Keras GPU | 89.34% | 4.37 pp | 719,034 | 8.484 | 2.456 |
| Memory | Keras GPU | 84.43% | 9.28 pp | 719,034 | 3.239 | 2.451 |
| Energy | Keras GPU | 87.46% | 6.25 pp | 719,364 | 8.502 | 2.284 |
| PTQ INT8 | TFLite CPU | 87.84% | 5.87 pp | - | 0.937 | 0.663 |
| QAT INT8 | TFLite CPU | 84.49% | 9.22 pp | - | 0.937 | 0.662 |
| Mixed FP16 | TFLite CPU | 89.26% | 4.45 pp | - | 1.369 | 0.535 |
| Dynamic range | TFLite CPU | 88.61% | 5.10 pp | - | 0.838 | 0.973 |

## 4. Accuracy Preservation Analysis

- The latency-optimized model retains 95.34% of baseline accuracy while its
  model file is 67.38% smaller.
- The memory-optimized model retains 90.10% of baseline accuracy and reduces
  model file size by 87.55%. Its 9.28 percentage-point accuracy drop is the
  explicit trade-off for smaller storage and reduced parameter density.
- The energy model's full branch reaches 87.74% accuracy. The early-exit
  policy reaches 87.46% while routing 81 of 10,000 samples through the early
  classifier, an early-exit fraction of 0.81%.
- Mixed FP16 provides the strongest quantized accuracy at 89.26%.
- Dynamic-range quantization provides the smallest quantized file, 0.838 MiB,
  while retaining 88.61% accuracy.
- PTQ provides a better accuracy-size balance than the simplified QAT
  implementation: 87.84% versus 84.49%, with effectively the same file size.

The assignment specifies an 85% target for the Part 1 baseline, not a minimum
accuracy for each optimized variant. The memory and QAT results are therefore
reported as measured trade-offs rather than treated as missing numeric targets.

## 5. Memory Management Analysis

- Gradient checkpointing is implemented as a training-only activation
  recomputation wrapper. Inference latency is not expected to improve.
- GPU/CPU model sharding separates feature extraction from the classification
  head. It adds cross-device transfer overhead: approximately 3.03 ms versus
  2.46 ms for the non-sharded latency model.
- FP16 activation compression applies to the classification head and measures
  approximately 2.21 ms for single-image inference.
- Batch-size search measured 1, 8, 16, 32, and 64. Batch 64 produced the
  highest throughput at approximately 9,997 images/s.

These measurements are proxies on the development machine, not MCU or
smartphone measurements.

## 6. Recommended Configurations

- Best accuracy-size balance for a general edge target: mixed FP16.
- Smallest model when storage is the primary constraint: dynamic range.
- Best calibrated INT8 balance: PTQ INT8.
- Lowest-storage Keras architecture: memory-optimized model, accepting the
  measured accuracy loss.
- Early exit is useful only for applications that tolerate its confidence
  policy; its coverage on this CIFAR-10 test set is limited to 0.81%.

## 7. Output Files

- `optimized_models/latency_optimized.keras`
- `optimized_models/memory_optimized.keras`
- `optimized_models/energy_optimized.keras`
- `quantized_models/ptq_int8.tflite`
- `quantized_models/qat_int8.tflite`
- `quantized_models/mixed_precision.tflite`
- `quantized_models/dynamic_range.tflite`
- `performance_comparison.json`
- `performance_comparison.csv`
