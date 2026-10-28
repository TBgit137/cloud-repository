# MobileNetV2 CIFAR-10 Hardware-Aware Optimization

This project implements a CIFAR-10 image classifier and evaluates how model
architecture, quantization, memory-management techniques, and execution
environment affect accuracy, latency, model size, and estimated energy.

The baseline is an ImageNet-pretrained MobileNetV2. The optimized variants
reduce input resolution and width, apply knowledge distillation and pruning,
introduce an early-exit classifier, and export several TFLite quantization
formats. Cross-platform evidence comes from native x86 execution, a simulated
Cortex-M system, a WebGPU proxy, and an analytical Cortex-A model.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── performance_profiler.py
├── optimization_framework.py
├── hardware_aware_analysis.pdf
├── performance_benchmarks.xlsx
├── optimization_guidelines.md
├── demo_notebook.ipynb
├── part1/
│   ├── part1_baseline_model.py
│   ├── baseline_mobilenetv2.keras
│   ├── README.md
│   └── part1_results/
├── part2/
│   ├── part2_optimizations.py
│   ├── part2_report.md
│   └── part2_results/
│       ├── optimized_models/
│       ├── quantized_models/
│       ├── performance_comparison.json
│       └── performance_comparison.csv
├── part3/
│   ├── part3_modeling.py
│   ├── part3_report.md
│   ├── README.md
│   ├── cortex_m/
│   ├── models/latency_optimized.onnx
│   ├── webgpu/
│   └── part3_results/
└── part4/
    ├── part4_analysis.py
    ├── README.md
    └── part4_results/
        ├── figures/
        ├── performance_report.json
        ├── hardware_utilization.json
        ├── design_cases.json
        └── CSV result tables
```

## Environment

The recorded runs used:

- Python 3.11
- TensorFlow and TFLite 2.15.1
- Keras 2.15.0
- NumPy 1.26.4
- Matplotlib 3.8.4
- QEMU ARM user mode 10.2.1
- Arm GNU Toolchain 14.3.Rel1 / GCC 14.3.1
- ONNX 1.16.2, ONNX Runtime 1.18.1, and tf2onnx 1.16.1
- ONNX Runtime Web 1.23.2
- NVIDIA GeForce RTX 4070 Ti for the recorded Keras runs

Create and activate a Python environment from the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On the first run, TensorFlow downloads CIFAR-10 and the ImageNet-pretrained
MobileNetV2 weights. Later runs reuse the local cache.

## 1. Baseline training

```bash
python part1/part1_baseline_model.py
```

The baseline uses a stratified 45,000/5,000 train-validation split of the
CIFAR-10 training set and evaluates on all 10,000 test images.

- Stage 1 trains the classification head for up to 8 epochs with the
  MobileNetV2 backbone frozen.
- Stage 2 unfreezes the non-BatchNorm backbone layers and fine-tunes for up to
  25 epochs at a lower learning rate.
- Training logs, metrics, and the saved model are written to `part1/`.

Recorded baseline results:

| Metric | Result |
|---|---:|
| Test accuracy | 93.71% |
| Parameters | 2,270,794 |
| FLOPs per image | 599,594,364 |
| Model file size | 26.008 MiB |
| Weight and BatchNorm memory | 8.662 MiB |
| Single-image latency | 2.497 ms |
| Batch-32 latency | 11.879 ms |

## 2. Optimized models

Train the optimized models and export the quantized variants:

```bash
python part2/part2_optimizations.py
```

To rebuild the comparison tables from existing artifacts without retraining:

```bash
python part2/part2_optimizations.py --reuse-existing
```

Implemented variants:

- Latency-optimized MobileNetV2 with `alpha=0.5` and 160x160 input.
- Memory-optimized model with width reduction, knowledge distillation, and
  masked magnitude pruning.
- Energy-oriented model with an early-exit classifier at `block_6_project`.
- PTQ INT8, QAT INT8, FP16 mixed precision, and dynamic-range quantization.

Recorded accuracy and model-size results:

| Model | Accuracy | Size (MiB) |
|---|---:|---:|
| Baseline | 93.71% | 26.008 |
| Latency optimized | 89.34% | 8.484 |
| Memory optimized | 84.43% | 3.239 |
| Energy optimized | 87.46% | 8.502 |
| PTQ INT8 | 87.84% | 0.937 |
| QAT INT8 | 84.49% | 0.937 |
| Mixed FP16 | 89.26% | 1.369 |
| Dynamic range | 88.61% | 0.838 |

## 3. Cross-platform analysis

Run the cross-platform modeling and validation pipeline:

```bash
python part3/part3_modeling.py
```

This command evaluates four execution contexts:

| Label | Platform and evidence |
|---|---|
| `[native-x86]` | Local TFLite measurement |
| `[QEMU-CM]` | Cortex-M3 QEMU with TFLite Micro |
| `[WebGPU-proxy]` | ONNX Runtime Web on a desktop GPU |
| `[modeled-ARM]` | Analytical Cortex-A78 estimate |

Recorded evidence includes:

- Native PTQ INT8: 0.663 ms.
- QEMU Cortex-M PTQ INT8: 946 ms.
- WebGPU proxy: 4.907 ms mean, 5.0 ms median, and 6.1 ms P95.
- Cortex-M model size: 982,152 bytes.
- Cortex-M tensor arena: 707,028 bytes.

The Cortex-M model fits the modeled 2 MiB flash budget, but the tensor arena
exceeds the modeled 512 KiB SRAM budget. QEMU demonstrates functional
execution but does not enforce physical SRAM limits.

### Cortex-M benchmark

Build the TensorFlow Lite Micro host library once if it is not already
available:

```bash
cd /home/mywslubuntu/tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_qemu TARGET_ARCH=cortex-m3 microlite
```

Build and run the TFLite Micro benchmark in QEMU:

```bash
cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
cd part3/cortex_m
make all
make run
```

If the TensorFlow Lite Micro checkout is not at
`/home/mywslubuntu/tflite-micro`:

```bash
make run TFLM_ROOT=/path/to/tflite-micro
```

### WebGPU proxy benchmark

Serve `part3/` over HTTP from the project root:

```bash
python -m http.server 8000 --directory part3
```

Open `http://localhost:8000/webgpu/benchmark.html`, wait for the benchmark to
finish, and close the server afterward. The page requires network access to
load ONNX Runtime Web from its CDN.

If the ONNX model needs to be rebuilt:

```bash
python -m tf2onnx.convert \
  --keras part2/part2_results/optimized_models/latency_optimized.keras \
  --output part3/models/latency_optimized.onnx \
  --opset 13
```

## 4. Comprehensive analysis

Generate the English report, design guidance, figures, and result tables:

```bash
python part4/part4_analysis.py
```

This command writes:

- `hardware_aware_analysis.md`
- `optimization_guidelines.md`
- JSON and CSV results under `part4/part4_results/`
- PNG figures under `part4/part4_results/figures/`

The existing `performance_benchmarks.xlsx` is not overwritten by the default
command. Use the following only when intentionally rebuilding that workbook:

```bash
python part4/part4_analysis.py --write-excel
```

`hardware_aware_analysis.md` is an intermediate source generated when the
analysis command is rerun. It is not required in the final submission package.

## Final outputs

- `hardware_aware_analysis.pdf`: analysis report generated from
  `hardware_aware_analysis.md`.
- `performance_benchmarks.xlsx`: detailed model, platform, Pareto, batch, and
  hardware-utilization data.
- `optimization_guidelines.md`: constraint-first design recommendations.
- `demo_notebook.ipynb`: interactive overview of model architecture,
  performance, Pareto fronts, and platform results.

To produce the PDF report, convert `hardware_aware_analysis.md` after checking
it. Keep `part4/part4_results/figures/` in place so the report's relative image
links resolve.

## Run the demo notebook

The notebook reads existing artifacts and does not retrain any model. Start
JupyterLab from the project root:

```bash
source /home/mywslubuntu/.venvs/maie5532/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Open the printed URL, select `demo_notebook.ipynb`, choose the `Python 3`
kernel associated with the activated environment, and run all cells.

Alternatively, open the project through VS Code Remote - WSL, open
`demo_notebook.ipynb`, select
`/home/mywslubuntu/.venvs/maie5532/bin/python` as the kernel, and run all
cells.

## Measurement notes

- Baseline and architecture-model latency use Keras on the recorded host.
- Quantized-model latency uses a single-threaded TFLite CPU interpreter.
- The WebGPU result is a desktop proxy, not a smartphone measurement.
- QEMU-CM is user-mode simulation and does not enforce physical SRAM, flash,
  cache, power, clock-frequency, or thermal limits.
- Energy values are analytical estimates derived from operation and memory
  counts; they are not power-meter measurements.
- Cache values in the utilization analysis are capacity proxies, not measured
  hit rates.
