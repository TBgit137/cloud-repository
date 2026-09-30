# Part 2: Cloud-Scale Optimization Report

## 1. Experimental Setup

The experiments used CIFAR-10, TensorFlow 2.15.1, and one NVIDIA GeForce RTX 4070 Ti. The data split contained 45,000 training images, 5,000 validation images, and 10,000 test images. Each configuration was trained for 50 epochs with seed 42. The checkpoint with the lowest validation loss was used for testing.

The ordinary student used the Part 1 architecture with 324,394 parameters. Student experiments started from the same initial weights and used the same learning-rate schedule. The Float32 model trained in Part 2, with 87.35% test accuracy, was the main reference. The Part 1 result was not used as the direct reference because its training schedule was different.

Total training time includes validation and checkpoint saving. Training throughput excludes the first epoch, validation, and saving. Peak memory means memory tracked by the TensorFlow GPU allocator, not total process GPU memory. One MiB equals 1,048,576 bytes. These are single-run comparisons; small differences should not be treated as guaranteed improvements.

## 2. Benchmark Results

| Configuration | Test accuracy | Training time (s) | Training throughput (images/s) | Peak GPU memory (MiB) |
|---|---:|---:|---:|---:|
| Float32, batch 128 | 87.35% | 141.74 | 17,690.1 | 410.56 |
| Mixed precision, batch 128 | 87.51% | 241.56 | 9,892.7 | 378.98 |
| Simulated distribution: one replica | 87.50% | 156.40 | 16,702.8 | 394.99 |
| Simulated distribution: two replicas | 87.55% | 547.48 | 4,452.4 | 212.80 / 215.35 |
| Serial data pipeline, batch 128 | 87.60% | 140.90 | 17,778.0 | 410.56 |
| Float32, batch 256 | 86.42% | 104.63 | 25,285.4 | 745.04 |
| Gradient accumulation, 64 × 4 | 87.55% | 307.73 | 7,688.3 | 319.23 |
| Teacher | 88.55% | 170.78 | 14,474.1 | 582.49 |
| Distilled student | 87.86% | 160.31 | 15,451.0 | 435.00 |

For the two-replica experiment, the memory values are reported separately for the two logical devices. They should not be added together to claim a total physical GPU memory measurement.

## 3. Comparison of Optimization Strategies

### 3.1 Mixed Precision

The model used mixed Float16 computation, a Float32 output layer, and dynamic loss scaling. Compared with Float32, test accuracy increased slightly from 87.35% to 87.51%, while peak memory decreased by 7.69%.

However, total training time increased from 141.74 to 241.56 seconds. Training throughput was only 0.56 times the Float32 throughput. Mean single-image inference latency was similar: 0.862 ms for Float32 and 0.850 ms for mixed precision. Saved model sizes were also similar, at 1.314 and 1.317 MiB, because mixed precision still kept model weights in Float32.

For this model and environment, mixed precision saved some memory but did not improve training speed.

### 3.2 Distributed Training Simulation

The experiment split one physical GPU into two logical devices, each with a 4096 MiB memory limit. The one-replica reference used the same device split. The two-replica model used MirroredStrategy with synchronous gradient aggregation. The global batch size stayed at 128.

Two replicas achieved 87.55% accuracy, compared with 87.50% for one replica. However, throughput fell from 16,702.8 to 4,452.4 images/s. The simulated throughput ratio was 0.267, and the ratio divided by two replicas was 13.33%.

This setup verified the distributed training process, but both replicas shared the same physical GPU. It did not provide additional physical computing resources. These results do not measure real two-GPU scaling efficiency.

### 3.3 Batch Processing

**Data pipeline.** Parallel data mapping and prefetching were compared with a serial pipeline at batch size 128. Throughput was 17,690.1 images/s for the optimized pipeline and 17,778.0 images/s for the serial pipeline, a difference of about 0.49%. The optimized pipeline showed no clear speed benefit in this run.

**Larger batches.** Increasing the batch size from 128 to 256 raised throughput by 42.94% and reduced total training time by 26.18%. However, peak memory increased from 410.56 to 745.04 MiB, and test accuracy fell by 0.93 percentage points. Larger batches improved speed at the cost of more memory and slightly lower accuracy under the tested schedule.

**Gradient accumulation.** Accumulating four micro-batches of 64 gave an effective batch size of 256. Compared with direct batch-256 training, peak memory decreased by 57.15%, from 745.04 to 319.23 MiB. Accuracy increased from 86.42% to 87.55%, but training took 307.73 seconds instead of 104.63 seconds. Both configurations made 8,800 optimizer updates. BatchNorm still used micro-batch statistics, so their training behavior was not exactly equivalent. Accumulation was useful for reducing memory use, but it was slower.

### 3.4 Knowledge Distillation

The teacher had 628,378 parameters, approximately 1.94 times the student's parameter count. The student kept the Part 1 architecture. Distillation used temperature 4 and a supervised-loss weight of 0.5, combining label cross-entropy with a temperature-scaled KL loss. The teacher was frozen during student training.

The teacher reached 88.55% accuracy. The ordinary student reached 87.35%, while the distilled student reached 87.86%. Distillation improved student accuracy by 0.51 percentage points, although it remained 0.69 percentage points below the teacher.

The distilled student retained the ordinary student's parameter count and 1.314 MiB model size; the teacher file was 2.474 MiB. Student distillation took 160.31 seconds, compared with 141.74 seconds for ordinary student training. Including teacher training, the total cost was 331.09 seconds. Distillation therefore gave a small observed accuracy improvement without increasing the deployed student size, but required extra training.

## 4. Conclusion

Batch size 256 gave the highest training throughput. Gradient accumulation reduced memory use but increased training time. Mixed precision saved a small amount of memory without a training speed benefit. Distillation improved student accuracy slightly while keeping its size unchanged. The distributed experiment demonstrated the synchronization setup only; real multi-GPU scaling would require separate physical GPUs.

Results source: `part2_results/cloud_optimization_report.json`, checked against each configuration's `metrics.json` and 50-epoch training history.
