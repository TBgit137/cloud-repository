# MobileNetV2 CIFAR-10 Baseline

This project trains a CIFAR-10 image classifier using an ImageNet-pretrained
MobileNetV2, a custom classification head, and two-stage transfer learning.

## Model and data

CIFAR-10 contains 50,000 training images and 10,000 test images across ten
classes. Ten percent of the training set is separated by a stratified split
for validation, leaving 45,000 training images and 5,000 validation images.
The original 32x32 inputs are resized to 224x224 with bilinear interpolation.
Training uses a batch size of 64, horizontal flips, small random rotation,
8% zoom, and 5% translation. Images remain in the 0-255 range until the model
rescales them to [-1, 1].

MobileNetV2 uses ImageNet weights without its original classification layer.
The project adds global average pooling, dropout with a rate of 0.2, and a
ten-class softmax layer.

## Environment

The recorded run used Python 3.11.16, TensorFlow 2.15.1, Keras 2.15.0, WSL2
Ubuntu, and an NVIDIA GeForce RTX 4070 Ti.
On the first run, TensorFlow downloads CIFAR-10 and the MobileNetV2 ImageNet
weights. Later runs reuse the local cache.

From the project root:

```bash
python3.11 -m venv ~/.venvs/maie5532
source ~/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

## Training and evaluation

```bash
python part1/part1_baseline_model.py
```

The script trains the classification head while MobileNetV2 is frozen, then
unfreezes the non-BatchNorm layers and fine-tunes with a lower learning rate.
It evaluates the final model on all 10,000 CIFAR-10 test images and benchmarks
single-image and batch inference.

Stage 1 runs for up to 8 epochs with a learning rate of `1e-3`. Stage 2 runs
for up to 25 epochs with a learning rate of `1e-4`. Both stages write epoch
logs and restore their best validation-accuracy weights at the end.

## Project files

| Path | Content |
|---|---|
| `baseline_mobilenetv2.keras` | Final trained model |
| `part1_results/training_stage1_log.csv` | Frozen-backbone training log |
| `part1_results/training_stage2_log.csv` | Fine-tuning log |
| `part1_results/baseline_metrics.json` | Accuracy and performance metrics |

## Results

| Metric | Result |
|---|---:|
| CIFAR-10 test accuracy | 93.71% |
| Parameters | 2,270,794 |
| FLOPs per image | 599,594,364 |
| `.keras` model file size | 26.0085 MiB |
| Model weight memory | 8.6624 MiB |
| Single-image mean latency | 2.4972 ms |
| Batch size 32 mean latency | 11.8789 ms |

Stage 1 completed 8 epochs and reached a best validation accuracy of 81.08%.
Stage 2 completed 25 epochs and reached a best validation accuracy of 94.02%
and a validation loss of 0.2051 at the final epoch. The final model uses the
best validation-accuracy weights from Stage 2.

The 26.0085 MiB `.keras` file contains the model and serialized training state.
The 8.6624 MiB memory figure covers TensorFlow model weights and BatchNorm
state only. These two values measure different things.

Latency includes the TensorFlow forward pass and output transfer to the host.
It excludes resizing, preprocessing, model loading, and disk access. The
single-image result is averaged over 100 runs and the batch-32 result over 40
runs, both after warm-up. Energy is not reported because no reliable
per-inference counter was available.
