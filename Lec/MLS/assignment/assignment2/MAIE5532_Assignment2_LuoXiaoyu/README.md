# MAIE5532 Assignment 2: Multi-Scale Optimization

Student: Luo Xiaoyu

This project optimizes a CIFAR-10 classifier for cloud, edge, and microcontroller deployment. It includes baseline training, cloud benchmarks, edge optimization, an integrated deployment pipeline, and a written analysis.

## 1. Environment and Installation

The experiments used Python 3.11, TensorFlow 2.15.1, WSL2 Ubuntu, and one NVIDIA GeForce RTX 4070 Ti with 12 GiB of GPU memory. Run the commands below inside WSL, from this submission directory (`MAIE5532_Assignment2_LuoXiaoyu`). GPU execution requires an NVIDIA driver accessible from WSL.

For a new environment with Python 3.11 available:

```bash
python3.11 -m venv ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, if Python is managed by uv:

```bash
uv python install 3.11
uv venv --python 3.11 --seed ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

If the environment already exists, activate it and install the requirements without recreating it. The root `requirements.txt` covers Parts 1–4. The CUDA extra is intended for Linux/WSL GPU execution.

Check the environment:

```bash
nvidia-smi
python -c "import tensorflow as tf; import tensorflow_model_optimization as tfmot; print('TensorFlow:', tf.__version__); print('TFMOT:', tfmot.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Expected versions are TensorFlow 2.15.1 and TFMOT 0.8.0. The GPU list should be nonempty for GPU benchmarks. CIFAR-10 is downloaded automatically on first use, which requires an internet connection.

## 2. Files and Outputs

| Location | Contents |
|---|---|
| `demo_notebook.ipynb` | Executable demonstration of saved results and example predictions |
| `example_inference.py` | Single-image inference for each final deployment target |
| `benchmark_deployment.py` | Final-model accuracy, size, and local inference timing benchmark |
| `part1/part1_baseline.py` | Baseline model and training |
| `part1/baseline_model.keras` | Trained baseline used by later parts |
| `part1/part1_results/` | Metrics, checkpoints, training history, CSV log, and training curves |
| `part2/part2_cloud_optimization.py` | Cloud optimization benchmarks |
| `part2/part2_results/cloud_optimized_models/` | Per-configuration models, metrics, and histories |
| `part2/part2_results/cloud_optimization_report.json` | Cloud benchmark summary |
| `part2/part2_results/performance_comparison.png` | Accuracy and training throughput comparison |
| `part3/part3_edge_optimization.py` | Pruning, quantization, lightweight architecture, and NAS |
| `part3/part3_results/edge_optimized_models/` | Pruned and lightweight Keras models |
| `part3/part3_results/tflite_models/` | Baseline and optimized TFLite models |
| `part3/part3_results/edge_optimization_report.json` | Edge benchmark summary |
| `part4/part4_deployment_pipeline.py` | Integrated model selection, conversion, and analysis |
| `part4/part4_results/optimized_models/` | Three final deployment models and quantized candidates |
| `part4/part4_results/multi_scale_optimization_report.json` | Multi-scale results, trade-offs, and recommendations |
| `part5/multi_scale_analysis.pdf` | Final written analysis |
| `part5/multi_scale_analysis.md` | Editable analysis source |

The scripts and model directories retain their required names within the part-specific folders. Submit this whole directory, including model files, notebook outputs, figures, and the final PDF. Exclude virtual environments, `__pycache__`, and `.ipynb_checkpoints` if they are recreated during local use.

Keep this directory structure: scripts locate earlier models and reports relative to their own files.

## 3. Execution

For a complete reproduction, run these commands in order:

```bash
python part1/part1_baseline.py
python part2/part2_cloud_optimization.py
python part3/part3_edge_optimization.py
python part4/part4_deployment_pipeline.py
```

Part 1 is required before Parts 2 and 3. Part 4 requires the complete Part 2 report and candidate models, plus the Part 3 report and lightweight models. If those saved outputs are already present, Part 4 can run directly without retraining earlier parts. Rerunning scripts overwrites their corresponding output files; preserve submitted results before reproducing experiments.

### Part 1: Baseline

The baseline uses three convolution blocks with BatchNorm, pooling, and a dense classifier. Training uses augmentation, early stopping, learning-rate reduction, and a validation checkpoint, with a maximum of 50 epochs. The data split is 45,000 training, 5,000 validation, and 10,000 test images.

The script prints the parameter count and test accuracy, then saves the model, metrics, and training curves. The recorded baseline reached 87.71% test accuracy with 324,394 parameters, exceeding the required 70%.

### Part 2: Cloud Optimization

The full run compares nine configurations covering Float32, mixed precision, one/two logical replicas, a serial data pipeline, batch size 256, gradient accumulation, teacher training, and distillation. The ordinary Float32 model is reused as the control. Configurations run in separate processes to isolate precision and device settings.

Individual experiment groups are available:

```bash
python part2/part2_cloud_optimization.py mixed
python part2/part2_cloud_optimization.py distributed
python part2/part2_cloud_optimization.py batch
python part2/part2_cloud_optimization.py distillation
```

Each group run replaces the summary with that group's results. Run the full benchmark to produce the complete report required by Part 4. Main settings are `EPOCHS=50`, `TEMPERATURE=4.0`, and `ALPHA=0.5` near the top of the script. The distributed experiment simulates two logical GPUs on one physical GPU; it does not measure real two-GPU scaling.

### Part 3: Edge Optimization

The script fine-tunes pruning targets of 50% and 75%, converts the baseline using dynamic range, full INT8, and Float16 quantization, trains a fixed lightweight architecture, and searches eight architecture candidates. It exports TFLite files and records accuracy, size, latency, sparsity, and deployment estimates.

The main settings are `PRUNE_EPOCHS=5`, `NAS_SEARCH_EPOCHS=4`, `ARCH_FINAL_EPOCHS=20`, `RESULT_EVAL_SAMPLES=10000`, and `LATENCY_RUNS=100`. INT8 calibration uses training data. The script prints the results and saves `edge_optimization_report.json`.

### Part 4: Multi-Scale Pipeline

The pipeline reuses Part 2 results to select the eligible non-simulated cloud model with the highest training throughput. It compares three baseline quantization variants using validation accuracy for the edge target, and converts the two Part 3 lightweight models to INT8 to select the smallest eligible microcontroller model. Final edge and microcontroller accuracy uses all 10,000 test images.

The three final files are `cloud_server.keras`, `edge_device.tflite`, and `microcontroller.tflite`. The report includes `deployment_targets`, `optimization_results`, `scaling_analysis`, and `deployment_recommendations`.

Recorded final results:

| Target | Selected strategy | Test accuracy | File size (MiB) |
|---|---|---:|---:|
| Cloud | Batch 256 | 86.42% | 1.314 |
| Edge | Dynamic range quantization | 87.67% | 0.324 |
| Microcontroller | Lightweight architecture + INT8 | 70.53% | 0.035 |

### Part 5: Analysis

The PDF summarizes optimization effectiveness, cross-scale trade-offs, resource constraints, development complexity, and deployment recommendations. No separate program is needed for Part 5.

## 4. Notebook Demo

`demo_notebook.ipynb` displays saved metrics, training curves, cloud comparisons, edge/NAS results, and final deployment results. It also runs one CIFAR-10 image through each final model. It does not retrain models.

From this submission directory, with the environment activated:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name maie5532 --display-name "Python (maie5532)"
python -m notebook demo_notebook.ipynb
```

Select the `Python (maie5532)` kernel, run all cells, and save the notebook to retain its outputs. Sections 1–5 read saved files; Section 6 runs example inference and downloads CIFAR-10 if it is not cached. Change `SAMPLE_INDEX` in Section 6 to select another test image. Tables and figures populate automatically; no manual screenshots are required. Predictions can differ between models and are not required to be correct for every image.

## 5. Example Inference and Deployment Benchmark

These scripts use the three final Part 4 models and the existing environment. No new dependencies or training are required. Run commands from the submission root after activating the environment.

Run inference for all targets on one CIFAR-10 test image:

```bash
python example_inference.py
```

Alternatively, select each target and an image index (0–9999):

```bash
python example_inference.py --target cloud_server --sample-index 0
python example_inference.py --target edge_device --sample-index 0
python example_inference.py --target microcontroller --sample-index 0
```

Output includes the true class, predicted class, output score, correctness, and local runtime. A wrong prediction for one image does not indicate a script failure. CIFAR-10 is downloaded if not cached.

First perform a quick benchmark check, then evaluate all 10,000 test images:

```bash
python benchmark_deployment.py --samples 100
python benchmark_deployment.py
```

To benchmark only one target, add `--target cloud_server`, `--target edge_device`, or `--target microcontroller`. The default is all three. The benchmark prints progress and records test accuracy, model file size, mean/median/p95 latency, and serial single-image inference throughput. Timing uses 20 warm-up calls and 100 measured calls, independently of the accuracy sample count.

Quick-check results are saved to `benchmark_results/deployment_benchmark_all_100.json`; full results are saved to `benchmark_results/deployment_benchmark_all_10000.json`. A single-target run uses its target name in the filename. Repeating the same command overwrites that benchmark file, without changing Part 1–4 results.

For the unchanged final models, full-test accuracy should be close to 86.42% (cloud), 87.67% (edge), and 70.53% (microcontroller). A 100-image subset need not match these values. Latency is newly measured and may differ from Part 4 because the inference wrapper and runtime conditions differ. All predictions use batch size 1; cloud output transfer is synchronized by conversion to NumPy. Inference memory and power remain `null` because these scripts do not measure them.

## 6. Interpreting Results

- Size fields use MiB (1,048,576 bytes). Model file size, weight memory, and runtime memory represent different quantities. The Part 1 Keras archive includes optimizer state; Part 2 deployment archives exclude it.
- Cloud throughput measures training, not inference. Training throughput excludes the first epoch, validation, and saving; total training time includes validation and saving. GPU memory values come from the TensorFlow allocator. Do not add the two logical-device peaks to claim physical GPU usage.
- Cloud latency uses local GPU inference with output transfer. TFLite latency uses local single-thread CPU inference with input/output handling. These are not direct cross-device speed comparisons.
- No physical edge board was tested. Part 3 RAM and energy values are rough assumptions. Part 4 records unmeasured target latency, inference memory, and power as `null`; this is expected, not a failed run.
- Recorded results are from individual runs. GPU execution and timing can vary when experiments are repeated.
