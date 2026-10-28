# MobileNetV2 CIFAR-10 Multi-Platform Analysis

This directory implements Track B with four clearly labelled execution
contexts:

| Context | Evidence type | Model and runtime |
|---|---|---|
| `[native-x86]` | Measured on the local x86_64 machine | PTQ INT8 with TFLite |
| `[QEMU-CM]` | Measured in user-mode Cortex-M3 QEMU | PTQ INT8 with TensorFlow Lite Micro |
| `[WebGPU-proxy]` | Measured in desktop Chromium/WebGPU | Float32 model with ONNX Runtime Web |
| `[modeled-ARM]` | Analytical estimate only | Roofline/energy model for Cortex-A78 |

The QEMU and WebGPU values are simulation/proxy evidence, not physical MCU or
smartphone measurements.

Verified tool versions:

| Tool | Version |
|---|---|
| Python | 3.11 |
| TensorFlow / TFLite | 2.15.1 |
| TensorFlow Lite Micro | commit `d0318206` |
| QEMU ARM user mode | 10.2.1 |
| Arm GNU Toolchain | 14.3.Rel1 / GCC 14.3.1 |
| ONNX / ONNX Runtime | 1.16.2 / 1.18.1 |
| ONNX Runtime Web | 1.23.2 |

## 1. Main analysis

From the project root:

```bash
source /home/mywslubuntu/.venvs/maie5532/bin/activate
python part3/part3_modeling.py
```

The command reads the Part 1/Part 2 artifacts, reruns the Cortex-M benchmark,
reads the saved WebGPU benchmark, produces analytical predictions, and writes:

- `part3_results/model_profiles.json`
- `part3_results/analytical_estimates.json`
- `part3_results/simulation_validation.json`
- `part3_results/cross_platform_analysis.json`
- `part3_results/qemu_cortex_m.json`
- `part3_report.md`

## 2. Cortex-M benchmark

The local repository already contains a statically linked benchmark. To
rebuild the TensorFlow Lite Micro host library once:

```bash
cd /home/mywslubuntu/tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_qemu TARGET_ARCH=cortex-m3 microlite
```

Then rebuild and run the benchmark:

```bash
cd assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu/part3/cortex_m
make run
```

If the TensorFlow Lite Micro checkout is elsewhere:

```bash
make run TFLM_ROOT=/path/to/tflite-micro
```

The firmware uses a 1 MiB tensor arena and reports both the configured arena
size and the actual amount used by TFLite Micro. QEMU user mode is suitable for
relative runtime comparison, but it does not enforce physical SRAM limits.

## 3. WebGPU proxy benchmark

The browser benchmark uses ONNX Runtime Web 1.23.2 and the exported
`models/latency_optimized.onnx`. Serve a directory that contains both
`part3/webgpu/` and `part3/models/` over HTTP, open
`webgpu/benchmark.html`, and wait for `Benchmark complete`.

```bash
cd assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
python -m http.server 8000 --directory part3
```

Open `http://localhost:8000/webgpu/benchmark.html`. The page loads ONNX
Runtime Web from a CDN, so network access is required.

The exported ONNX model can be regenerated with:

```bash
cd assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
source /home/mywslubuntu/.venvs/maie5532/bin/activate
python -m tf2onnx.convert \
  --keras part2/part2_results/optimized_models/latency_optimized.keras \
  --output part3/models/latency_optimized.onnx \
  --opset 13
```

Do not use `file://` for the WebGPU page because browser security policies
block loading the model across local files. Close the local HTTP server when
the benchmark is complete.

## 4. Interpretation limits

- Analytical values are raw, uncalibrated estimates. They are used to test
  whether a simple Roofline model is sufficient, not as substitutes for
  measurements.
- QEMU user mode does not model the Cortex-M7 cache, bus, power, thermal state,
  or the physical 512 KiB SRAM limit.
- WebGPU on a desktop GPU is a relative mobile-GPU proxy. Browser scheduling
  and desktop drivers differ from a smartphone.
- The energy values are estimates from operation counts and must not be
  reported as measured energy.
