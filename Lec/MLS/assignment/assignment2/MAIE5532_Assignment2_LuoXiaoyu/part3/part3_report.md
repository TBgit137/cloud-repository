# Part 3: Edge Deployment Optimization Report

## 1. Experimental Setup

The experiments used the trained Part 1 CIFAR-10 model with 324,394 parameters. The data split contained 45,000 training images, 5,000 validation images, and 10,000 test images, with seed 42. Pruning used 5 epochs of fine-tuning. Each NAS candidate was trained for 4 epochs, while the fixed lightweight architecture and the selected NAS architecture were trained from scratch for 20 epochs.

All reported test accuracies used 10,000 images. TensorFlow Lite inference ran on the local WSL CPU with one interpreter thread and batch size 1. Mean latency was measured over 100 runs after 20 warm-up runs. Timing includes the Python inference wrapper, input/output handling, and quantization conversions where needed; image normalization is excluded. No physical edge device was available. One MiB equals 1,048,576 bytes, and one KiB equals 1,024 bytes.

## 2. Pruning and Sparsity Analysis

Magnitude-based pruning used a polynomial schedule from zero sparsity toward targets of 50% and 75%. The model was fine-tuned with Adam at a learning rate of 0.0001 during pruning, and pruning wrappers were removed before export.

| Model | Recorded zero-value ratio | Keras test accuracy | Parameters | Weight memory (MiB) | Gzip Keras size (bytes) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.00% | 87.70% | 324,394 | 1.237 | 3,572,324 |
| Pruning target 50% | 49.61% | 87.64% | 324,394 | 1.237 | 732,258 |
| Pruning target 75% | 74.42% | 84.60% | 324,394 | 1.237 | 436,089 |

The recorded ratio counts zeros across all stored model weights, including biases and BatchNorm state, rather than only prunable kernels. It therefore differs from the target kernel sparsity. Weight memory excludes activations, optimizer state, and runtime overhead.

The 50% target preserved accuracy closely, while the 75% target reduced accuracy by 3.10 percentage points relative to the Keras baseline. Both retained the original parameter count and dense weight memory. The 75% pruned Keras archive compressed to 40.45% fewer bytes than the 50% version. However, the baseline archive includes restored optimizer state, while the stripped models were recompiled with fresh optimizers, so its gzip size is not a controlled measure of pruning savings.

Both pruned Float32 TFLite files remained 1.238 MiB. Their mean latencies were 0.378 and 0.407 ms, compared with 0.382 ms for the baseline. Thus, pruning made the Keras files more compressible but did not reduce dense TFLite storage or show a consistent inference speed improvement.

## 3. Quantization and TensorFlow Lite Results

The baseline was converted using dynamic range, full INT8, and Float16 quantization. Full INT8 calibration used 100 training images and integer input/output tensors. The pruned and lightweight models were also exported as Float32 TFLite models.

| TFLite model | Test accuracy | File size (MiB) | Mean latency (ms) |
|---|---:|---:|---:|
| Baseline Float32 | 87.71% | 1.238 | 0.382 |
| Dynamic range | 87.67% | 0.324 | 0.319 |
| Full INT8 | 87.39% | 0.328 | 0.333 |
| Float16 | 87.72% | 0.625 | 0.367 |
| Pruning target 50%, Float32 | 87.65% | 1.238 | 0.378 |
| Pruning target 75%, Float32 | 84.57% | 1.238 | 0.407 |
| Lightweight architecture, Float32 | 70.54% | 0.065 | 0.039 |
| NAS-selected architecture, Float32 | 67.96% | 0.141 | 0.097 |

Dynamic range quantization reduced file size by 73.81%, losing only 0.04 percentage points of accuracy. It had the lowest local latency among the baseline quantization variants. Full INT8 reduced size by 73.52% with a 0.32-point accuracy loss. Float16 reduced size by 49.46% and essentially preserved accuracy. Its 0.01-point increase represents only one test image and is not evidence of a meaningful improvement.

Float32 TFLite conversion preserved Keras accuracy within 0.03 percentage points for the baseline and pruned models, and matched both lightweight models at the reported precision. All eight TFLite files were generated in `part3_results/tflite_models/`. Size comparisons use these deployment files rather than Keras training archives.

## 4. Architecture Optimization and NAS

The lightweight architecture replaced standard convolutions with depthwise separable convolutions, used ReLU6 in convolution blocks, replaced global average pooling with global max pooling, and reduced channel widths. The fixed configuration used depth 3, width multiplier 0.5, and 3 × 3 kernels.

It contained 15,157 parameters, a 95.33% reduction from the baseline. Its weight memory was 60,628 bytes, and its TFLite file was 67,880 bytes, 94.77% smaller than the baseline. Local inference was approximately 9.82 times faster, but accuracy fell from 87.71% to 70.54%. These results describe the combined architecture changes, not the individual contribution of each change.

NAS evaluated eight combinations of depth, width, and kernel size. The following validation accuracies are the best values during each 4-epoch search run. A candidate is on the accuracy–parameter Pareto frontier if no other candidate has at least equal accuracy and no more parameters, with a strict improvement in at least one measure.

| Depth | Width multiplier | Kernel size | Parameters | Validation accuracy | Pareto frontier |
|---|---:|---|---:|---:|---|
| 2 | 0.5 | 3 × 3 | 5,589 | 50.48% | Yes |
| 2 | 0.5 | 5 × 5 | 6,661 | 45.58% | No |
| 2 | 0.75 | 3 × 3 | 11,245 | 53.40% | Yes |
| 2 | 0.75 | 5 × 5 | 12,829 | 57.98% | Yes |
| 3 | 0.5 | 3 × 3 | 15,157 | 57.34% | No |
| 3 | 0.5 | 5 × 5 | 17,765 | 55.22% | No |
| 3 | 0.75 | 3 × 3 | 31,741 | 61.20% | Yes |
| 3 | 0.75 | 5 × 5 | 35,629 | 62.10% | Yes |

The frontier labels above were derived from the saved results. The program selected the highest score, defined as validation accuracy minus parameter count divided by 1,000,000, rather than explicitly filtering the frontier. Its selected configuration, depth 3 / width 0.75 / kernel 5, is on this frontier. Parameter count serves as an efficiency proxy during search, not a latency measurement.

After 20 epochs of fresh training, the selected architecture achieved 67.96% test accuracy with 35,629 parameters. The fixed lightweight model achieved higher accuracy with fewer parameters and lower measured latency. The short search therefore did not identify the best final model in this run.

## 5. Edge Deployment Viability

Deployment feasibility was assessed using model storage budgets because no edge board was available. The first, third, and fourth budgets below come from the program. A second hypothetical microcontroller budget is included to compare two microcontroller resource levels; it is a storage-only calculation using the same exported files, not an additional hardware experiment.

| Assumed platform budget | Model storage budget | RAM budget | Models fitting the model storage budget |
|---|---:|---:|---|
| Microcontroller A | 1 MiB | 256 KiB | Three quantized baseline variants and both lightweight architectures |
| Microcontroller B | 256 KiB | 64 KiB | Both lightweight architectures only |
| Small Linux edge device | 64 MiB | 64 MiB | All eight models |
| Raspberry Pi-class budget | 512 MiB | 512 MiB | All eight models |

These budgets are assumed comparison scenarios, not specifications of named boards. Storage fit excludes firmware. The program approximates RAM as twice the model file size, but this does not measure activation buffers or runtime memory. Consequently, its `estimated_ram_ok` values do not establish actual RAM feasibility. Target-device latency and operator support also remain unverified.

The program estimates energy using `energy (mJ) = assumed power (mW) × local latency (ms) / 1000`. The table gives the resulting scenario values for three representative models.

| Model | 50 mW scenario (mJ) | 1,000 mW scenario (mJ) | 3,000 mW scenario (mJ) |
|---|---:|---:|---:|
| Baseline Float32 | 0.01908 | 0.38169 | 1.14506 |
| Dynamic range | 0.01595 | 0.31908 | 0.95723 |
| Lightweight architecture | 0.00194 | 0.03886 | 0.11658 |

These are illustrative energy calculations assuming the local CPU latency is retained at each power level. They are not measured or validated predictions of microcontroller or edge-board energy consumption.

For the tested local runtime, dynamic range quantization provided a strong size–accuracy balance. The fixed lightweight architecture was the smallest and fastest option, with a substantial accuracy cost. It is the strongest storage-constrained candidate among the tested models, but actual device deployment still depends on RAM, supported operations, and target latency.

Results source: `part3_results/edge_optimization_report.json`; TFLite file sizes checked against the exported files.
