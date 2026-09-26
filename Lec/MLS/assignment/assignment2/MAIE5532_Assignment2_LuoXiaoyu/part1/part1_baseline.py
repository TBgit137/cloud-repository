"""第一部分：按照作业给定的函数框架实现 CIFAR-10 基线模型。

在作业根目录运行：python part1/part1_baseline.py
依赖：tensorflow、numpy、matplotlib。
模型和实验结果保存在本脚本所在的 part1 目录中。
"""

import json
import os
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras


SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 50
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "part1_results"


def create_baseline_model():
    """创建并编译卷积神经网络。"""
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        # 卷积块 1：特征图尺寸由 32×32 缩小为 16×16。
        keras.layers.Conv2D(32, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(32, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        # 卷积块 2：特征图尺寸由 16×16 缩小为 8×8。
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        # 卷积块 3：特征图尺寸由 8×8 缩小为 4×4。
        keras.layers.Conv2D(128, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, (3, 3), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((2, 2)),
        # 分类器：输出类别概率，与稀疏分类交叉熵损失相匹配。
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ], name="cifar10_baseline")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data():
    """加载并返回归一化后的 CIFAR-10 数组；数据增强在训练阶段应用。

    此处不进行数据增强，便于划分未经增强的验证集，
    同时保留题目规定的四个数组返回值。
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)


def _make_training_datasets(x_train, y_train):
    """在数据增强前，从训练集的每个类别中划出 10% 作为验证集。"""
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for label in np.unique(y_train):
        indices = rng.permutation(np.flatnonzero(y_train == label))
        count = max(1, int(len(indices) * 0.1))
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    train_indices = rng.permutation(train_indices)
    val_indices = np.asarray(val_indices)

    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1,
            fill_mode="reflect", seed=SEED + 1,
        ),
    ], name="training_augmentation")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[train_indices], y_train[train_indices])
    )
    train_ds = train_ds.shuffle(len(train_indices), seed=SEED).batch(BATCH_SIZE)
    # 每轮训练重新生成随机增强；验证集和测试集不进行数据增强。
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[val_indices], y_train[val_indices])
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def _measure_inference(model, x_test, repetitions=200):
    """预热后测量批大小为 1 的前向推理耗时，包含结果传回主机的时间。

    不包含数据预处理和模型加载时间。通过 numpy() 取出结果，
    确保设备计算完成后再停止计时。
    """
    @tf.function(input_signature=[tf.TensorSpec((1, 32, 32, 3), tf.float32)])
    def infer(images):
        return model(images, training=False)

    samples = tf.convert_to_tensor(x_test[:repetitions])
    for _ in range(20):
        infer(samples[:1]).numpy()
    durations_ms = []
    for index in range(repetitions):
        sample = samples[index:index + 1]
        start = time.perf_counter()
        infer(sample).numpy()
        durations_ms.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": 1,
        "warmup_runs": 20,
        "measured_runs": repetitions,
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "scope": "forward pass plus host output transfer; excludes preprocessing",
    }


def train_baseline_model(model, x_train, y_train, x_test, y_test):
    """最多训练 50 轮，返回最佳模型、训练历史和评估指标。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = _make_training_datasets(x_train, y_train)
    checkpoint_path = RESULTS_DIR / "best_baseline.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / "training_log.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - start
    # 显式加载最佳检查点，确保达到最大训练轮数时也使用最佳模型。
    model = keras.models.load_model(checkpoint_path)
    test_metrics = model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE, verbose=0, return_dict=True,
    )
    weight_bytes = sum(
        int(np.prod(weight.shape)) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )
    metrics = {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "accuracy_requirement_met": bool(test_metrics["accuracy"] > 0.70),
        "parameters": int(model.count_params()),
        "weight_memory_bytes": weight_bytes,
        "weight_memory_mib": weight_bytes / (1024 ** 2),
        "memory_scope": "model weights including BatchNorm state; excludes activations, optimizer, and runtime",
        "training_seconds": training_seconds,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "inference": _measure_inference(model, x_test),
        "tensorflow_version": tf.__version__,
        "visible_devices": [device.name for device in tf.config.get_visible_devices()],
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "validation_fraction": 0.1,
    }
    return model, history, metrics


def _save_history(history):
    """保存训练历史，并绘制训练集与验证集的准确率和损失曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {key: [float(v) for v in series] for key, series in history.history.items()}
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(values, indent=2), encoding="utf-8",
    )
    epochs = range(1, len(values["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(epochs, values[metric], label="Training")
        axis.plot(epochs, values[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = create_baseline_model()
    model.summary()
    model, history, metrics = train_baseline_model(
        model, x_train, y_train, x_test, y_test,
    )
    model_path = OUTPUT_DIR / "baseline_model.keras"
    model.save(model_path)
    metrics["model_size_bytes"] = os.path.getsize(model_path)
    metrics["model_size_mib"] = metrics["model_size_bytes"] / (1024 ** 2)
    metrics["model_file_scope"] = "Keras archive including optimizer state"
    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    _save_history(history)
    print(f"Baseline model parameters: {model.count_params():,}")
    print(f"Baseline test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Results saved to: {RESULTS_DIR}")
    if not metrics["accuracy_requirement_met"]:
        print("Requirement not met: test accuracy must be strictly greater than 70%.")
