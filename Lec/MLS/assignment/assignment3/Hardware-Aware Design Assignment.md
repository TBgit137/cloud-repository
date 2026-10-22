# Hardware-Aware Design Assignment

**Due:** Week 8

## Objective

Design and implement a hardware-aware machine learning application that considers energy, latency, and memory constraints across different deployment targets. You will optimize a computer vision model for multiple hardware platforms, implementing hardware-specific optimizations and analyzing performance trade-offs through either real hardware deployment or comprehensive simulation and modeling.

## Background

This assignment builds on the hardware acceleration principles discussed in weeks 6-8. You will experience firsthand how hardware characteristics influence ML system design, from algorithm selection to memory management to energy optimization. The assignment emphasizes the practical challenges of deploying AI at the edge while maintaining acceptable performance.

## Hardware Requirements Notice

**This assignment can be completed entirely using simulation tools and performance modeling. Students with access to specialized hardware are encouraged to use it, but it is not required for full credit.**

You may choose between:

- **Track A: Real Hardware Deployment** - Deploy on actual hardware platforms with real measurements.
- **Track B: Simulation & Modeling** - Use simulation tools and theoretical analysis for comprehensive platform evaluation.

Both tracks can achieve full credit and meet all learning objectives.

## Part 1: Baseline Model Implementation (25 points)

Implement a MobileNetV2-based image classification model that will serve as your baseline for hardware-aware optimizations.

### Implementation Requirements

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import psutil
import os


def create_baseline_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a MobileNetV2-based model for image classification.

    Args:
        input_shape: Input image dimensions
        num_classes: Number of classification classes

    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # TODO: Implement MobileNetV2 architecture
    # Base model: MobileNetV2 (pretrained on ImageNet, frozen initially)
    # Add custom classification head for your dataset
    # Include global average pooling and dropout for regularization
    pass


def load_and_preprocess_data():
    """
    Load CIFAR-10 dataset and preprocess for MobileNetV2.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    # TODO: Load CIFAR-10 dataset
    # Resize images to 224x224 for MobileNetV2
    # Normalize pixel values and apply data augmentation
    # Use tf.data.Dataset for efficient data loading
    pass


def benchmark_baseline_model(model, test_data, batch_size=32):
    """
    Benchmark the baseline model for latency and memory usage.

    Args:
        model: Trained Keras model
        test_data: Test dataset
        batch_size: Batch size for inference

    Returns:
        dict: Performance metrics
    """
    # TODO: Measure inference latency (single sample and batch)
    # TODO: Monitor memory usage during inference
    # TODO: Calculate FLOPs and model parameters
    # TODO: Measure energy consumption (if possible on your platform)

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
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Create and train model
    model = create_baseline_model()

    # Train with fine-tuning strategy
    # 1. Train classification head only (base frozen)
    # 2. Unfreeze base model and fine-tune with lower learning rate

    # Save the trained model
    model.save('baseline_mobilenetv2.keras')

    # Benchmark performance
    metrics = benchmark_baseline_model(model, x_test)
    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
```

### Deliverables for Part 1

- Complete baseline MobileNetV2 implementation.
- Training logs showing the transfer learning approach.
- Achieved test accuracy (target: >85% on CIFAR-10).
- Baseline performance metrics including latency, memory, and model size.

## Part 2: Hardware-Aware Optimizations (30 points)

Implement multiple optimization techniques targeting different hardware constraints.

### 2.1 Model Architecture Optimization

```python
def create_optimized_models():
    """
    Create hardware-optimized variants of the baseline model.

    Returns:
        dict: Models optimized for different constraints
    """
    models = {}

    # TODO: Implement latency-optimized model
    # - Reduce MobileNetV2 depth multiplier
    # - Use depthwise separable convolutions
    # - Optimize for inference speed
    models['latency_optimized'] = create_latency_optimized_model()

    # TODO: Implement memory-optimized model
    # - Reduce model width (channels)
    # - Use model pruning techniques
    # - Optimize for memory footprint
    models['memory_optimized'] = create_memory_optimized_model()

    # TODO: Implement energy-optimized model
    # - Use quantization-aware training
    # - Reduce computation intensity
    # - Optimize for energy efficiency
    models['energy_optimized'] = create_energy_optimized_model()

    return models


def create_latency_optimized_model():
    """Create model optimized for inference latency."""
    # TODO: Use MobileNetV2 with alpha=0.5 (depth multiplier)
    # TODO: Reduce input resolution to 160x160 or 128x128
    # TODO: Use grouped convolutions where possible
    pass


def create_memory_optimized_model():
    """Create model optimized for memory usage."""
    # TODO: Implement model pruning (structured and unstructured)
    # TODO: Use knowledge distillation from baseline model
    # TODO: Reduce number of filters per layer systematically
    pass


def create_energy_optimized_model():
    """Create model optimized for energy consumption."""
    # TODO: Implement quantization-aware training
    # TODO: Use early exit mechanisms
    # TODO: Optimize activation functions for efficiency
    pass
```

### 2.2 Quantization Implementation

```python
def apply_quantization_optimizations(model, x_train_sample):
    """
    Apply different quantization techniques to the model.

    Args:
        model: Trained Keras model
        x_train_sample: Representative dataset for calibration

    Returns:
        dict: Quantized model variants
    """
    quantized_models = {}

    # TODO: Post-training quantization (PTQ)
    quantized_models['ptq_int8'] = post_training_quantization(model, x_train_sample)

    # TODO: Quantization-aware training (QAT)
    quantized_models['qat_int8'] = quantization_aware_training(model, x_train_sample)

    # TODO: Mixed precision quantization
    quantized_models['mixed_precision'] = mixed_precision_quantization(model)

    # TODO: Dynamic range quantization
    quantized_models['dynamic_range'] = dynamic_range_quantization(model)

    return quantized_models


def representative_dataset_generator(x_train_sample):
    """Generator for quantization calibration."""
    # TODO: Implement representative dataset for calibration
    # Should yield 100-500 samples covering data distribution
    pass
```

### 2.3 Memory Management Optimization

```python
def implement_memory_optimizations(model):
    """
    Implement memory-aware optimizations.

    Args:
        model: Base model to optimize

    Returns:
        dict: Memory optimization results
    """
    optimizations = {}

    # TODO: Implement gradient checkpointing
    optimizations['gradient_checkpointing'] = implement_gradient_checkpointing(model)

    # TODO: Implement model sharding for large models
    optimizations['model_sharding'] = implement_model_sharding(model)

    # TODO: Implement activation compression
    optimizations['activation_compression'] = implement_activation_compression(model)

    # TODO: Optimize batch size for memory constraints
    optimizations['optimal_batch_size'] = find_optimal_batch_size(model)

    return optimizations
```

### Deliverables for Part 2

- Implementation of all four optimization strategies.
- Quantized model variants with accuracy preservation analysis.
- Memory optimization techniques with measured improvements.
- Performance comparison across all optimized variants.

## Part 3: Multi-Platform Analysis (25 points)

Choose **Track A** OR **Track B** based on your hardware access.

### Track A: Real Hardware Deployment

Deploy your optimized models across different hardware platforms and measure real performance.

#### 3.1 Platform-Specific Optimization

```python
class HardwareOptimizer:
    """Hardware-aware model optimizer for different platforms."""

    def __init__(self, target_platform):
        self.platform = target_platform
        self.optimization_config = self._get_platform_config()

    def _get_platform_config(self):
        """Get optimization configuration for target platform."""
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
        """Apply platform-specific optimizations."""
        # TODO: Implement platform-specific optimization pipeline
        # CPU: Use Intel OpenVINO or ARM Compute Library optimizations
        # ARM Cortex-A: Use NEON SIMD instructions, FP16
        # ARM Cortex-M: Use CMSIS-NN, aggressive quantization
        # Mobile GPU: Use GPU delegates, texture memory optimization
        pass


# TODO: Implement platform-specific deployment
def deploy_to_tflite_micro(model, target_mcu='cortex_m4'):
    """Deploy model to TensorFlow Lite Micro for MCU."""
    pass


def deploy_to_mobile_gpu(model, target_gpu='adreno_640'):
    """Deploy model using GPU acceleration."""
    pass


def deploy_to_edge_tpu(model):
    """Deploy model to Google Coral Edge TPU."""
    pass
```

#### Track A Deliverables

- Working deployments on at least 3 different platform types.
- Platform-specific optimization implementations.
- Real performance measurements across all platforms.
- Stress testing results showing sustained performance.

### Track B: Simulation & Performance Modeling

Develop comprehensive performance models and validate using simulation tools.

#### 3.1 Platform Performance Modeling

```python
class PlatformPerformanceModel:
    """Model performance characteristics across different hardware platforms."""

    def __init__(self, platform_specs):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_performance_models(self):
        """Build analytical performance models for each platform."""
        models = {}

        # TODO: Implement roofline models for each platform
        # Consider memory bandwidth, compute throughput, cache hierarchy
        models['roofline'] = self._build_roofline_model()

        # TODO: Implement energy models based on operation counts
        models['energy'] = self._build_energy_model()

        # TODO: Implement memory access models
        models['memory'] = self._build_memory_model()

        return models

    def estimate_performance(self, model_graph, platform_type):
        """Estimate performance metrics for a model on a platform."""
        # TODO: Analyze model graph for FLOPs, memory access patterns
        # TODO: Apply platform-specific performance models
        # TODO: Consider thermal throttling and power constraints
        pass


def validate_with_simulation(models, platform_configs):
    """Validate performance estimates using simulation tools."""

    validation_results = {}

    # TODO: Use QEMU for ARM simulation
    validation_results['qemu_arm'] = simulate_arm_performance(models)

    # TODO: Use Renode for Cortex-M simulation
    validation_results['renode_cm'] = simulate_cortex_m_performance(models)

    # TODO: Use WebGPU as mobile GPU proxy
    validation_results['webgpu_proxy'] = simulate_mobile_gpu_performance(models)

    # TODO: Cross-validate simulation results with analytical models
    validation_results['model_accuracy'] = cross_validate_models(validation_results)

    return validation_results


def simulate_arm_performance(models):
    """Simulate ARM Cortex-A performance using QEMU."""
    # TODO: Set up QEMU ARM environment
    # TODO: Deploy TFLite models and measure performance
    # TODO: Extract detailed performance metrics
    pass


def simulate_cortex_m_performance(models):
    """Simulate Cortex-M performance using Renode."""
    # TODO: Set up Renode simulation environment
    # TODO: Deploy TFLite Micro models
    # TODO: Measure cycle counts and memory usage
    pass


def simulate_mobile_gpu_performance(models):
    """Simulate mobile GPU performance using WebGPU proxy."""
    # TODO: Convert models to WebGPU-compatible format
    # TODO: Measure performance in browser environment
    # TODO: Scale results based on mobile GPU specifications
    pass
```

#### 3.2 Cross-Platform Analysis Framework

```python
class CrossPlatformAnalyzer:
    """Analyze optimization effectiveness across different platforms."""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()

    def _load_platform_specs(self):
        """Load detailed specifications for target platforms."""
        return {
            'x86_desktop': {
                'core_count': 8,
                'frequency_ghz': 3.5,
                'simd_width': 256,  # AVX2
                'l1_cache_kb': 32,
                'l2_cache_kb': 256,
                'l3_cache_mb': 8,
                'memory_bandwidth_gbps': 50,
                'tdp_watts': 65
            },
            'arm_cortex_a78': {
                'core_count': 4,
                'frequency_ghz': 2.4,
                'simd_width': 128,  # NEON
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
        """Analyze how different optimizations perform across platforms."""

        results = {}

        for platform in self.platform_characteristics:
            platform_results = {}

            # TODO: Model baseline performance on platform
            baseline_perf = self.model_performance(baseline_model, platform)

            # TODO: Model optimized variants performance
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
        """Model expected performance of a model on a platform."""
        # TODO: Implement detailed performance modeling
        # Consider instruction-level analysis, memory access patterns
        # Cache behavior, thermal constraints
        pass
```

#### Track B Deliverables

- Comprehensive performance models for 3+ platform types.
- Simulation validation using multiple tools (QEMU, Renode, WebGPU).
- Cross-platform optimization effectiveness analysis.
- Detailed theoretical analysis with literature validation.

## Part 4: Comprehensive Analysis and Design Recommendations (20 points)

### 4.1 Performance Analysis

Create detailed performance comparisons and provide hardware-aware design recommendations.

```python
def generate_performance_report():
    """Generate comprehensive performance analysis report."""

    # TODO: Create performance comparison tables
    performance_data = {
        'models': ['baseline', 'latency_opt', 'memory_opt', 'energy_opt'],
        'platforms': ['cpu_x86', 'arm_cortex_a', 'arm_cortex_m', 'mobile_gpu'],
        'metrics': ['latency_ms', 'memory_mb', 'energy_mj', 'accuracy', 'throughput_fps']
    }

    # TODO: Generate radar charts for multi-dimensional trade-offs
    # TODO: Create Pareto frontier analysis for accuracy vs efficiency
    # TODO: Analyze performance scaling with different batch sizes

    return performance_data


def analyze_hardware_utilization():
    """Analyze how well each model utilizes different hardware features."""

    utilization_analysis = {
        'cpu_simd_utilization': {},  # SIMD instruction usage
        'memory_bandwidth_efficiency': {},  # Memory access patterns
        'cache_efficiency': {},  # L1/L2 cache hit rates
        'thermal_characteristics': {},  # Temperature profiles
        'power_efficiency': {}  # Energy per operation
    }

    return utilization_analysis
```

### 4.2 Design Methodology

```python
class HardwareAwareDesignMethodology:
    """Framework for hardware-aware ML system design."""

    def __init__(self):
        self.design_principles = [
            'co_design_hardware_software',
            'early_constraint_specification',
            'iterative_optimization',
            'multi_objective_optimization',
            'platform_specific_tuning'
        ]

    def constraint_specification_framework(self, application_requirements):
        """Framework for specifying hardware constraints early in design."""
        # TODO: Define constraint specification methodology
        # Power budget, memory limits, latency requirements, accuracy targets
        pass

    def optimization_priority_framework(self, constraints):
        """Determine optimization priorities based on constraints."""
        # TODO: Create decision tree for optimization strategy selection
        pass

    def design_space_exploration(self, base_model, constraints):
        """Systematic exploration of design alternatives."""
        # TODO: Implement automated design space exploration
        # Architecture variants, quantization levels, optimization techniques
        pass
```

## Required Analysis Report (3-4 pages)

### 1. Hardware-Software Co-Design Analysis

- How did hardware constraints influence your model architecture decisions?
- What trade-offs did you observe between accuracy and hardware efficiency?
- How did different optimization techniques interact with hardware characteristics?

### 2. Platform-Specific Optimization Insights

- Which optimizations were most effective for each platform type?
- How did memory hierarchy differences impact optimization strategies?
- What role did specialized hardware features (SIMD, tensor cores) play in performance?

### 3. Energy-Latency-Accuracy Trade-off Analysis

- Create Pareto frontier plots showing trade-offs between metrics.
- Analyze which applications would benefit from each optimization approach.
- Discuss the implications for battery-powered edge devices.

### 4. Scalability and Deployment Considerations

- How do your optimizations scale across different hardware generations?
- What challenges arise when deploying across heterogeneous hardware?
- How would you handle model updates in resource-constrained environments?

### 5. Design Methodology Recommendations

- Propose a systematic approach for hardware-aware ML system design.
- What tools and frameworks would improve the hardware-aware design process?
- How should hardware constraints be incorporated into the ML development lifecycle?

### 6. Future Hardware Trends Impact

- How might emerging hardware trends (edge AI chips, neuromorphic computing) affect your design decisions?
- What new optimization opportunities do you foresee?

## Submission Requirements

### Code Deliverables

- `part1_baseline_model.py` - Baseline MobileNetV2 implementation.
- `part2_optimizations.py` - Hardware-aware optimizations.
- `part3_deployment.py` (Track A) OR `part3_modeling.py` (Track B) - Platform analysis code.
- `performance_profiler.py` - Performance measurement utilities.
- `optimization_framework.py` - Hardware-aware design framework.
- `requirements.txt` - Python dependencies.
- `Makefile` or build scripts for deployments/simulations.

### Model Files

- `baseline_mobilenetv2.keras` - Trained baseline model.
- `optimized_models/` - Directory containing all optimized model variants.
- Platform-specific model files (`.tflite`, `.onnx`, etc.).

### Analysis Documents

- `hardware_aware_analysis.pdf` - Comprehensive analysis report (3-4 pages).
- `performance_benchmarks.xlsx` - Detailed performance measurements.
- `optimization_guidelines.md` - Design methodology documentation.

### Demo Materials

- `README.md` - Complete setup and execution instructions.
- `demo_notebook.ipynb` - Jupyter notebook demonstrating key results.
- Platform-specific demo scripts.
- Video demonstration (optional, 3-5 minutes).

## Hardware Access and Simulation Resources

### Track A: Recommended Hardware Platforms

- **CPU:** Intel/AMD x86, ARM Cortex-A series.
- **Mobile GPU:** Qualcomm Adreno, ARM Mali, Apple GPU.
- **Edge AI:** Google Coral Dev Board, Intel Neural Compute Stick.
- **MCU:** Arduino Nano 33 BLE, STM32 Discovery boards.

### Track B: Simulation Alternatives

If physical hardware is unavailable, use:

- **ARM Cortex-A (Linux-class):** QEMU (system or user mode) to run ARM binaries (e.g., TFLite/ONNX Runtime) under an ARM root filesystem. Label results as `[QEMU-ARM]`.
- **ARM Cortex-M (MCU-class):** Renode (preferred) or QEMU Cortex-M to run TensorFlow Lite Micro firmware with INT8 models; report cycles/latency from simulator logs. Label as `[Renode-CM]` or `[QEMU-CM]`.
- **Mobile GPU proxy:** ONNX Runtime Web with WebGPU in a desktop Chromium-based browser, or ncnn (Vulkan) on desktop as a proxy for mobile GPUs. Label as `[WebGPU-proxy]` or `[Vulkan-proxy]`.
- **x86 CPU:** Native desktop/laptop execution using TFLite or ONNX Runtime for baseline comparisons. Label as `[native-x86]`.

### Notes for Track B

- Clearly tag all reported metrics with the execution context (e.g., `[QEMU-ARM]`).
- Document versions, commands, and flags used so results are reproducible.
- Treat simulated/proxy results as relative comparisons; discuss limitations in your report.
- Clearly label any numbers derived from estimation as "estimated" and state the method.
- Prefer empirical measurements (even in simulators) over purely theoretical calculations.

### Performance Estimation Tools (Both Tracks)

- **TensorFlow Model Optimization Toolkit (TF-MOT):** Analyze effects of pruning/quantization.
- **TFLite `benchmark_model` and `onnxruntime_perf_test`:** Run repeatable latency tests.
- **FLOPs/params estimators:** `tensorflow.python.profiler`, `keras-flops`, or Netron.
- **Power/energy tools:** `powertop`/`turbostat` (Linux) for package power estimation.
- **Published benchmarks:** Reference vendor/community results for comparative context.
