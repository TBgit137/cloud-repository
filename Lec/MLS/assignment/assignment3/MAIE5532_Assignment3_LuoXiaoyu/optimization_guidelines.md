# Hardware-Aware Optimization Guidelines

## 1. Define constraints before choosing a model

Record the deployment platform, execution label, accuracy floor, latency
ceiling, energy budget, model-size ceiling, and SRAM or working-set ceiling.
Reject a candidate that violates any hard constraint before comparing its
secondary metrics.

## 2. Use a staged optimization order

1. Profile the baseline with the deployment runtime.
2. Reduce input resolution, width, or graph work.
3. Apply INT8 PTQ when the operator set and accelerator support integer paths.
4. Use QAT only when calibration and training time are sufficient.
5. Add pruning or early exit when the workload distribution supports it.
6. Re-measure memory, latency, accuracy, and energy on the target runtime.

## 3. Match optimization to hardware pressure

- **x86 CPU:** use SIMD-aware runtimes and compare against cached, warmed-up
  single-image and batched execution.
- **ARM Cortex-A:** prioritize reduced FLOPs, FP16 or INT8, and bandwidth-aware
  layouts.
- **Cortex-M:** verify flash and SRAM separately. An artifact that fits in flash
  can still fail because of TFLite Micro's tensor arena.
- **Mobile/WebGPU:** amortize launch overhead with sufficient parallel work;
  do not expect small-model changes to reduce end-to-end latency linearly.

## 4. Use Pareto analysis rather than a single score

The measured Pareto set for accuracy, latency, and model size is:

**baseline, latency_optimized, ptq_int8, qat_int8, mixed_precision, dynamic_range.**

Choose from this set according to the application. Use the highest-accuracy
model when errors are costly, the smallest or fastest model when resource
budgets dominate, and the balanced model when deployment requirements do not
strongly prioritize one dimension.

## 5. Validate before deployment

Use the complete test set for accuracy, not calibration samples. Confirm all
operators exist in the target runtime. Measure sustained performance after
warm-up and thermal stabilization. Record model hashes and runtime versions.
Keep a rollback artifact on the device.

## Reference design cases

| Case | Platform | Feasible models | Recommendation |
|---|---|---|---|
| accuracy-sensitive mobile deployment | arm_cortex_a78 | latency_optimized, mixed_precision, dynamic_range | latency_optimized |
| strict SRAM MCU deployment | arm_cortex_m7 | None | No fully feasible model |
