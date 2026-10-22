# Part 1: MobileNetV2 Baseline

This directory implements the CIFAR-10 baseline for the hardware-aware design
assignment. The model uses ImageNet-pretrained MobileNetV2 at a 224 x 224 input
resolution, a custom classification head, and two-stage transfer learning.

## Environment

The recorded run used:

- Python 3.11.16
- TensorFlow 2.15.1
- Keras 2.15.0
- WSL2 Ubuntu
- NVIDIA GeForce RTX 4070 Ti

From the submission directory, create a Python 3.11 environment and install
the dependencies:

```bash
cd MAIE5532_Assignment3_LuoXiaoyu
python3.11 -m venv ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

The first training run requires network access to download the CIFAR-10
dataset and MobileNetV2 ImageNet weights.

## Training

Run from the submission directory:

```bash
python part1/part1_baseline_model.py
```

The default configuration uses:

- Batch size: 64
- Stratified validation split: 10% of the original CIFAR-10 training set
- Stage 1: frozen MobileNetV2, Adam with a learning rate of `1e-3`
- Stage 2: unfrozen MobileNetV2 except BatchNorm layers, Adam with a learning
  rate of `1e-4`
- Stage 1 and Stage 2 epoch limits: 8 and 25
- Benchmark batch size: 32
- Single-image and batch inference runs: 100 and 40
- Warm-up runs: 20

Training and evaluation use `tf.data.Dataset`. Images retain their original
0-255 range in the data pipeline, and a `Rescaling` layer inside the model maps
them to the `[-1, 1]` range expected by MobileNetV2. Data augmentation is
applied only to the training split. The CIFAR-10 test split is used only for
the final evaluation.

## Benchmark and Output Files

The default command trains the model, evaluates it on all 10,000 test images,
and saves:

| Path | Content |
|---|---|
| `baseline_mobilenetv2.keras` | Final trained model |
| `part1_results/training_stage1_log.csv` | Stage 1 epoch log |
| `part1_results/training_stage2_log.csv` | Stage 2 epoch log |
| `part1_results/training_history.json` | Training history for both stages |
| `part1_results/training_manifest.json` | Completed epochs and run configuration |
| `part1_results/training_curves.png` | Training and validation curves |
| `part1_results/model_summary.txt` | Model summary and parameter count |
| `part1_results/baseline_metrics.json` | Accuracy, latency, memory, size, and FLOPs |
| `part1_results/stage1_best.keras` | Best Stage 1 validation checkpoint |
| `part1_results/stage2_best.keras` | Best Stage 2 validation checkpoint |

The saved model can be evaluated again without retraining:

```bash
python part1/part1_baseline_model.py --benchmark-only
```

This command rewrites `baseline_metrics.json` with a new timing run.

## Recorded Results

| Metric | Result |
|---|---:|
| CIFAR-10 test accuracy | 93.71% |
| Parameters | 2,270,794 |
| FLOPs per image | 599,594,364 |
| Model file size | 9.1617 MiB |
| Single-image mean latency | 2.2894 ms |
| Batch size 32 mean latency | 11.1140 ms |
| Batch size 32 throughput | 2,879.25 images/s |
| TensorFlow GPU allocator peak | 616.5 MiB |

Stage 1 completed 4 epochs and selected a best validation accuracy of 80.84%.
Stage 2 completed 14 epochs and selected a best validation accuracy of 94.30%.
The final model is the best Stage 2 validation checkpoint; the configured
epoch limits were 8 and 25.

Latency was measured in WSL2 on the local GPU and includes the TensorFlow
forward pass and transfer of the output to the host. It excludes image resize,
preprocessing, model loading, and disk access. Timing can vary between runs.
Energy per inference is recorded as `null` because no reliable per-inference
energy counter was available for the WSL GPU process.
