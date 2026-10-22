# Part 1: MobileNetV2 Baseline

This directory contains the CIFAR-10 baseline required by Part 1. It uses an
ImageNet-pretrained MobileNetV2, a custom classification head, and two-stage
transfer learning.

## Environment

The recorded run used Python 3.11.16, TensorFlow 2.15.1, Keras 2.15.0, WSL2
Ubuntu, and an NVIDIA GeForce RTX 4070 Ti.

From the `MAIE5532_Assignment3_LuoXiaoyu` directory:

```bash
python3.11 -m venv ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python part1/part1_baseline_model.py
```

The script trains the classification head while MobileNetV2 is frozen, then
unfreezes the non-BatchNorm layers and fine-tunes with a lower learning rate.
It evaluates the final model on all 10,000 CIFAR-10 test images and benchmarks
single-image and batch inference.

## Outputs

| Path | Content |
|---|---|
| `baseline_mobilenetv2.keras` | Final trained model |
| `part1_results/training_stage1_log.csv` | Frozen-backbone training log |
| `part1_results/training_stage2_log.csv` | Fine-tuning log |
| `part1_results/baseline_metrics.json` | Accuracy and performance metrics |

## Recorded Results

| Metric | Result |
|---|---:|
| CIFAR-10 test accuracy | 93.71% |
| Parameters | 2,270,794 |
| FLOPs per image | 599,594,364 |
| Model size | 9.1617 MiB |
| Single-image mean latency | 2.2894 ms |
| Batch size 32 mean latency | 11.1140 ms |
| Model weight memory | 8.6624 MiB |

Stage 1 completed four epochs and Stage 2 completed fourteen epochs. The final
model uses the best validation weights from the fine-tuning stage.

Latency includes the TensorFlow forward pass and output transfer to the host.
It excludes resizing, preprocessing, model loading, and disk access. Energy is
recorded as `null` because no reliable per-inference counter was available.
