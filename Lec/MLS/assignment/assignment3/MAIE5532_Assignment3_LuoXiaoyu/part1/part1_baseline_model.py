#!/usr/bin/env python3
"""第 1 部分：MobileNetV2 图像分类基线模型。"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


SEED = 42
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_STAGE1_EPOCHS = 8
DEFAULT_STAGE2_EPOCHS = 25
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-4
ACCURACY_TARGET = 0.85

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "part1_results"
MODEL_PATH = OUTPUT_DIR / "baseline_mobilenetv2.keras"
STAGE1_CHECKPOINT = RESULTS_DIR / "stage1_best.keras"
STAGE2_CHECKPOINT = RESULTS_DIR / "stage2_best.keras"
TRAINING_MANIFEST = RESULTS_DIR / "training_manifest.json"


def configure_runtime(seed: int = SEED) -> None:
    """设置随机种子并允许 GPU 按需分配显存。"""
    keras.utils.set_random_seed(seed)
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            # 初始化后无法再修改显存增长设置；此时继续使用当前配置。
            pass


def create_baseline_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    创建一个用于图像分类的 MobileNetV2 基础模型。

    参数：
        input_shape：输入图像尺寸
        num_classes：分类类别数

    返回：
        tf.keras.Model：已编译、可直接训练的模型
    """
    # 基础模型：MobileNetV2（在 ImageNet 上预训练，初始阶段冻结）
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_input")
    # MobileNetV2 的 ImageNet 权重期望输入范围为 [-1, 1]。
    # 缩放层放在模型内部，使保存后的模型直接接受 0-255 像素值。
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_preprocessing",
    )(inputs)
    x = base_model(x, training=False)

    # 为你的数据集添加自定义分类头
    # 使用全局平均池化和 dropout 进行正则化
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.2, name="classifier_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cifar10_mobilenetv2_baseline",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data():
    """
    加载 CIFAR-10 数据集，并为 MobileNetV2 进行预处理。

    返回：
        tuple：(x_train, y_train, x_test, y_test)
    """
    # 加载 CIFAR-10 数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # 保留 0-255 的像素范围。模型内的 Rescaling 层统一执行 [-1, 1] 归一化，
    # 从而避免“先除以 255，再使用 MobileNetV2 预处理”造成的重复缩放。
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.reshape(-1).astype(np.int64)
    y_test = y_test.reshape(-1).astype(np.int64)
    return x_train, y_train, x_test, y_test


def _stratified_train_validation_split(
    labels: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """按类别分层划分训练集和验证集，测试集始终不参与训练。"""
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        validation_count = max(1, int(round(len(class_indices) * validation_fraction)))
        validation_indices.append(class_indices[:validation_count])
        train_indices.append(class_indices[validation_count:])

    train_indices_array = rng.permutation(np.concatenate(train_indices))
    validation_indices_array = rng.permutation(np.concatenate(validation_indices))
    return train_indices_array, validation_indices_array


def _make_augmentation() -> keras.Sequential:
    """创建只用于训练集的轻度图像增强。"""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(
                factor=0.03,
                fill_mode="nearest",
                seed=SEED + 1,
            ),
            layers.RandomZoom(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="nearest",
                seed=SEED + 2,
            ),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="nearest",
                seed=SEED + 3,
            ),
        ],
        name="training_augmentation",
    )


def _resize_images(images: tf.Tensor) -> tf.Tensor:
    """将 CIFAR-10 图像缩放为 MobileNetV2 所需尺寸。"""
    images = tf.cast(images, tf.float32)
    return tf.image.resize(images, IMAGE_SIZE, method="bilinear")


def _make_train_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    augmentation: keras.Sequential,
) -> tf.data.Dataset:
    """构建训练集流水线，并在在线缩放前应用数据增强。"""

    def prepare(images_batch: tf.Tensor, labels_batch: tf.Tensor):
        images_batch = tf.cast(images_batch, tf.float32)
        images_batch = augmentation(images_batch, training=True)
        images_batch = tf.image.resize(
            images_batch,
            IMAGE_SIZE,
            method="bilinear",
        )
        return images_batch, labels_batch

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(
        buffer_size=len(images),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _make_eval_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    """构建不打乱、不增强的验证或测试集流水线。"""

    def prepare(images_batch: tf.Tensor, labels_batch: tf.Tensor):
        return _resize_images(images_batch), labels_batch

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    validation_fraction: float,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """划分数据并创建训练、验证和测试流水线。"""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须位于 0 和 1 之间。")

    # 使用独立随机数生成器，保证数据划分可复现。
    rng = np.random.default_rng(SEED)
    train_indices, validation_indices = _stratified_train_validation_split(
        y_train,
        validation_fraction=validation_fraction,
        rng=rng,
    )

    augmentation = _make_augmentation()
    train_dataset = _make_train_dataset(
        x_train[train_indices],
        y_train[train_indices],
        batch_size=batch_size,
        augmentation=augmentation,
    )
    validation_dataset = _make_eval_dataset(
        x_train[validation_indices],
        y_train[validation_indices],
        batch_size=batch_size,
    )
    test_dataset = _make_eval_dataset(
        x_test,
        y_test,
        batch_size=batch_size,
    )
    return train_dataset, validation_dataset, test_dataset


def _callbacks_for_stage(
    stage_name: str,
    checkpoint_path: Path,
    early_stopping_patience: int,
    learning_rate_patience: int,
) -> list[keras.callbacks.Callback]:
    """创建两阶段训练共用的日志、检查点和学习率回调。"""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=learning_rate_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(RESULTS_DIR / f"training_{stage_name}_log.csv"),
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def _get_mobilenet_backbone(model: keras.Model) -> keras.Model:
    """从外层模型中定位 MobileNetV2 基础模型。"""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2") and hasattr(layer, "layers"):
            return layer
    raise ValueError("无法在模型中定位 MobileNetV2 基础模型。")


def train_baseline_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    stage1_epochs: int,
    stage2_epochs: int,
) -> tuple[keras.Model, keras.callbacks.History, keras.callbacks.History]:
    """执行“冻结分类头训练 + 低学习率微调”的两阶段迁移学习。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 使用微调策略进行训练
    # 1. 仅训练分类头（冻结基础模型）
    base_model = _get_mobilenet_backbone(model)
    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history_stage1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=stage1_epochs,
        callbacks=_callbacks_for_stage(
            stage_name="stage1",
            checkpoint_path=STAGE1_CHECKPOINT,
            early_stopping_patience=3,
            learning_rate_patience=2,
        ),
        verbose=2,
    )

    # 从第一阶段的验证集最佳权重继续，而不是从最后一个 epoch 继续。
    model = keras.models.load_model(STAGE1_CHECKPOINT, compile=False)
    base_model = _get_mobilenet_backbone(model)

    # 2. 解冻基础模型，并使用较低学习率进行微调
    base_model.trainable = True
    # 保留预训练 BatchNorm 的移动统计量，避免小验证集使统计量明显偏移。
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history_stage2 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=stage2_epochs,
        callbacks=_callbacks_for_stage(
            stage_name="stage2",
            checkpoint_path=STAGE2_CHECKPOINT,
            early_stopping_patience=6,
            learning_rate_patience=3,
        ),
        verbose=2,
    )

    model = keras.models.load_model(STAGE2_CHECKPOINT, compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, history_stage1, history_stage2


def _save_training_history_artifacts(
    history_stage1: dict[str, list[float]],
    history_stage2: dict[str, list[float]],
    stage1_epochs: int,
    stage2_epochs: int,
) -> None:
    """保存两阶段历史记录和带阶段分界线的训练曲线。"""
    history = {
        "stage1": history_stage1,
        "stage2": history_stage2,
        "configured_max_epochs": {
            "stage1": stage1_epochs,
            "stage2": stage2_epochs,
        },
    }
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage1_length = len(history_stage1["loss"])
    stage2_length = len(history_stage2["loss"])
    stage1_epochs_axis = np.arange(1, stage1_length + 1)
    stage2_epochs_axis = np.arange(
        stage1_length + 1,
        stage1_length + stage2_length + 1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(
            stage1_epochs_axis,
            history_stage1[metric],
            label=f"Stage 1 {metric}",
        )
        axis.plot(
            stage1_epochs_axis,
            history_stage1[f"val_{metric}"],
            linestyle="--",
            label=f"Stage 1 validation {metric}",
        )
        axis.plot(
            stage2_epochs_axis,
            history_stage2[metric],
            label=f"Stage 2 {metric}",
        )
        axis.plot(
            stage2_epochs_axis,
            history_stage2[f"val_{metric}"],
            linestyle="--",
            label=f"Stage 2 validation {metric}",
        )
        axis.axvline(
            stage1_length + 0.5,
            color="black",
            linestyle=":",
            linewidth=1,
            label="Fine-tuning starts",
        )
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)

    fig.suptitle("MobileNetV2 transfer learning on CIFAR-10")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def _save_training_artifacts(
    history_stage1: keras.callbacks.History,
    history_stage2: keras.callbacks.History,
    stage1_epochs: int,
    stage2_epochs: int,
) -> None:
    """把 Keras History 转换为普通字典后保存训练证据。"""
    _save_training_history_artifacts(
        {
            key: [float(value) for value in values]
            for key, values in history_stage1.history.items()
        },
        {
            key: [float(value) for value in values]
            for key, values in history_stage2.history.items()
        },
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
    )


def _read_training_csv(path: Path) -> dict[str, list[float]]:
    """从 Keras CSVLogger 输出恢复可绘图的训练历史。"""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"训练日志为空：{path}")
    columns = [column for column in rows[0] if column != "epoch"]
    return {
        column: [float(row[column]) for row in rows]
        for column in columns
    }


def _write_training_manifest(
    *,
    run_mode: str,
    batch_size: int,
    validation_fraction: float,
    stage1_configured_epochs: int,
    stage2_configured_epochs: int,
    stage1_completed_epochs: int,
    stage2_completed_epochs: int,
    training_seconds: float | None,
    completion_note: str,
) -> None:
    """写入实际训练配置和完成情况，供基准测试与报告引用。"""
    TRAINING_MANIFEST.write_text(
        json.dumps(
            {
                "run_mode": run_mode,
                "batch_size": batch_size,
                "validation_fraction": validation_fraction,
                "stage1_configured_epochs": stage1_configured_epochs,
                "stage2_configured_epochs": stage2_configured_epochs,
                "stage1_completed_epochs": stage1_completed_epochs,
                "stage2_completed_epochs": stage2_completed_epochs,
                "stage1_learning_rate": STAGE1_LEARNING_RATE,
                "stage2_learning_rate": STAGE2_LEARNING_RATE,
                "training_seconds": training_seconds,
                "seed": SEED,
                "image_size": list(IMAGE_SIZE),
                "accuracy_target": ACCURACY_TARGET,
                "completion_note": completion_note,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _model_weight_memory_bytes(model: keras.Model) -> int:
    """计算模型权重和 BatchNorm 状态的存储字节数。"""
    return sum(
        int(np.prod(weight.shape)) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )


def _estimate_flops(model: keras.Model) -> int | None:
    """使用 TensorFlow profiler 估算单张图像的浮点运算次数。"""
    try:
        concrete_function = tf.function(
            lambda images: model(images, training=False),
        ).get_concrete_function(
            tf.TensorSpec((1, *INPUT_SHAPE), tf.float32),
        )
        frozen_function = convert_variables_to_constants_v2(concrete_function)
        profiler_options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        profiler_options["output"] = "none"
        profile = tf.compat.v1.profiler.profile(
            graph=frozen_function.graph,
            run_meta=tf.compat.v1.RunMetadata(),
            cmd="op",
            options=profiler_options,
        )
        return int(profile.total_float_ops)
    except Exception:
        # FLOPs 的统计实现不参与训练；不同 TensorFlow/Keras 版本下失败时保留可识别状态。
        return None


def _gpu_memory_snapshot() -> dict[str, float | None]:
    """读取 TensorFlow 当前可见 GPU 的分配器内存；CPU 环境返回空值。"""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {"gpu_memory_current_mb": None, "gpu_memory_peak_mb": None}
    try:
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        return {
            "gpu_memory_current_mb": memory_info["current"] / (1024 ** 2),
            "gpu_memory_peak_mb": memory_info["peak"] / (1024 ** 2),
        }
    except (ValueError, KeyError):
        return {"gpu_memory_current_mb": None, "gpu_memory_peak_mb": None}


def _timing_summary(timings_ms: list[float]) -> dict[str, float]:
    """汇总平均、中位数和 P95 延迟。"""
    values = np.asarray(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def benchmark_baseline_model(
    model: keras.Model,
    test_data: tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    *,
    model_path: Path | None = None,
    inference_runs: int = 100,
    batch_inference_runs: int = 40,
    warmup_runs: int = 20,
) -> dict[str, object]:
    """
    对基线模型的延迟和内存使用情况进行基准测试。

    参数：
        model：已训练的 Keras 模型
        test_data：测试数据集
        batch_size：推理时使用的批大小

    返回：
        dict：性能指标
    """
    x_test, y_test = test_data
    test_dataset = _make_eval_dataset(x_test, y_test, batch_size=batch_size)

    # 先在完整测试集上计算准确率和损失。
    evaluation = model.evaluate(
        test_dataset,
        verbose=0,
        return_dict=True,
    )

    # 测量单样本和批量推理延迟
    single_count = min(max(1, inference_runs), len(x_test))
    batch_count = min(max(1, batch_inference_runs), max(1, len(x_test) // batch_size))
    single_batch = _resize_images(x_test[:single_count])
    timing_batch = _resize_images(x_test[:batch_size])

    @tf.function(input_signature=[tf.TensorSpec((1, *INPUT_SHAPE), tf.float32)])
    def infer_single(images):
        return model(images, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec((batch_size, *INPUT_SHAPE), tf.float32)],
    )
    def infer_batch(images):
        return model(images, training=False)

    # 预热 TensorFlow 计算图，防止首次编译时间进入正式延迟统计。
    for _ in range(warmup_runs):
        infer_single(single_batch[:1]).numpy()
        infer_batch(timing_batch).numpy()

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    gpu_before = _gpu_memory_snapshot()

    single_timings_ms: list[float] = []
    for index in range(single_count):
        sample = single_batch[index:index + 1]
        start = time.perf_counter()
        infer_single(sample).numpy()
        single_timings_ms.append((time.perf_counter() - start) * 1000.0)

    batch_timings_ms: list[float] = []
    for _ in range(batch_count):
        start = time.perf_counter()
        infer_batch(timing_batch).numpy()
        batch_timings_ms.append((time.perf_counter() - start) * 1000.0)

    rss_after = process.memory_info().rss
    gpu_after = _gpu_memory_snapshot()
    single_timing = _timing_summary(single_timings_ms)
    batch_timing = _timing_summary(batch_timings_ms)

    parameters = int(model.count_params())
    weight_memory_bytes = _model_weight_memory_bytes(model)
    model_size_bytes = model_path.stat().st_size if model_path else None

    metrics = {
        "single_inference_time": single_timing["mean_ms"],
        "batch_inference_time": batch_timing["mean_ms"],
        "memory_usage_mb": weight_memory_bytes / (1024 ** 2),
        "model_size_mb": (
            model_size_bytes / (1024 ** 2)
            if model_size_bytes is not None
            else None
        ),
        "accuracy": float(evaluation["accuracy"]),
        "flops": _estimate_flops(model),
        "parameters": parameters,
        "test_loss": float(evaluation["loss"]),
        "accuracy_target": ACCURACY_TARGET,
        "accuracy_requirement_met": bool(evaluation["accuracy"] > ACCURACY_TARGET),
        "single_inference_time_unit": "ms",
        "batch_inference_time_unit": "ms",
        "single_inference_runs": single_count,
        "batch_inference_runs": batch_count,
        "warmup_runs": warmup_runs,
        "single_latency": single_timing,
        "batch_latency": batch_timing,
        "batch_size": batch_size,
        "batch_throughput_samples_per_second": (
            batch_size * 1000.0 / batch_timing["mean_ms"]
        ),
        "model_weight_memory_mb": weight_memory_bytes / (1024 ** 2),
        "process_rss_delta_mb": max(0.0, (rss_after - rss_before) / (1024 ** 2)),
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "memory_usage_scope": (
            "TF variable memory including BatchNorm state; excludes activations, "
            "optimizer state, shared libraries, and runtime allocators"
        ),
        "model_size_scope": (
            ".keras archive size as written by Keras; it may include serialized "
            "optimizer state in addition to model weights"
            if model_size_bytes is not None
            else "not measured"
        ),
        "latency_scope": (
            "TensorFlow forward pass plus host output transfer; excludes image "
            "resize, preprocessing, model loading, and disk access"
        ),
        "energy_j_per_inference": None,
        "energy_status": (
            "Not measured: no reliable per-inference energy counter was available "
            "for the WSL GPU process."
        ),
        "tensorflow_version": tf.__version__,
        "keras_version": importlib.metadata.version("keras"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "visible_devices": [
            device.name for device in tf.config.get_visible_devices()
        ],
        "seed": SEED,
    }
    return metrics


def _save_model_summary(model: keras.Model) -> None:
    """把模型摘要保存为文本，便于在没有 Jupyter 的环境中查看。"""
    lines: list[str] = []
    model.summary(print_fn=lines.append, line_length=120)
    (RESULTS_DIR / "model_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_and_save_benchmark(
    model: keras.Model,
    test_data: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
    *,
    run_start: float,
    training_seconds: float | None,
    run_mode: str,
) -> dict[str, object]:
    """执行基准测试、写入指标文件并打印结果。"""
    metrics = benchmark_baseline_model(
        model,
        test_data,
        batch_size=32,
        model_path=MODEL_PATH,
        inference_runs=args.inference_runs,
        batch_inference_runs=args.batch_inference_runs,
    )
    metrics["run_mode"] = run_mode
    metrics["run_elapsed_seconds"] = time.perf_counter() - run_start
    metrics["training_seconds"] = training_seconds
    metrics["benchmark_invocation"] = {
        "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage1_learning_rate": STAGE1_LEARNING_RATE,
        "stage2_learning_rate": STAGE2_LEARNING_RATE,
    }
    if TRAINING_MANIFEST.exists():
        metrics["training_run"] = json.loads(
            TRAINING_MANIFEST.read_text(encoding="utf-8"),
        )
    else:
        metrics["training_run"] = None
    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Results saved to: {RESULTS_DIR}")
    return metrics


def _parse_args() -> argparse.Namespace:
    """解析训练和基准测试参数。"""
    parser = argparse.ArgumentParser(
        description="训练并评估 assignment 3 的 MobileNetV2 基线模型。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="训练和评估使用的批大小。",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="从原始训练集按类别划分出的验证集比例。",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=DEFAULT_STAGE1_EPOCHS,
        help="第一阶段冻结基础模型时最多训练的 epoch 数。",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=DEFAULT_STAGE2_EPOCHS,
        help="第二阶段低学习率微调时最多训练的 epoch 数。",
    )
    parser.add_argument(
        "--inference-runs",
        type=int,
        default=100,
        help="单样本延迟正式测量的次数。",
    )
    parser.add_argument(
        "--batch-inference-runs",
        type=int,
        default=40,
        help="批量延迟正式测量的次数。",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="只训练和保存模型，不执行完整性能基准测试。",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="加载已保存的 baseline_mobilenetv2.keras，只重新执行性能基准测试。",
    )
    parser.add_argument(
        "--finalize-from-checkpoint",
        action="store_true",
        help=(
            "训练中断时，从 stage2_best.keras 和 CSV 日志补齐最终模型、"
            "训练证据和基准测试。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """执行完整的数据加载、两阶段训练、保存和性能评估流程。"""
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    if args.stage1_epochs <= 0 or args.stage2_epochs <= 0:
        raise ValueError("两个阶段的 epoch 数都必须大于 0。")
    if args.benchmark_only and args.finalize_from_checkpoint:
        raise ValueError(
            "--benchmark-only 与 --finalize-from-checkpoint 不能同时使用。"
        )

    configure_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    if args.benchmark_only:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"找不到已训练模型：{MODEL_PATH}。请先运行完整训练。"
            )
        model = keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        _save_model_summary(model)
        metrics = _run_and_save_benchmark(
            model,
            (x_test, y_test),
            args,
            run_start=run_start,
            training_seconds=None,
            run_mode="benchmark_only",
        )
        if not metrics["accuracy_requirement_met"]:
            raise SystemExit(
                f"Test accuracy {metrics['accuracy']:.4f} did not exceed "
                f"the required {ACCURACY_TARGET:.2f}."
            )
        return

    if args.finalize_from_checkpoint:
        stage1_log = RESULTS_DIR / "training_stage1_log.csv"
        stage2_log = RESULTS_DIR / "training_stage2_log.csv"
        if not STAGE2_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"找不到第二阶段最佳 checkpoint：{STAGE2_CHECKPOINT}"
            )
        if not stage1_log.exists() or not stage2_log.exists():
            raise FileNotFoundError("补齐最终结果需要两个阶段的 CSV 训练日志。")

        history_stage1 = _read_training_csv(stage1_log)
        history_stage2 = _read_training_csv(stage2_log)
        model = keras.models.load_model(STAGE2_CHECKPOINT, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        _save_model_summary(model)
        model.save(MODEL_PATH)
        _save_training_history_artifacts(
            history_stage1,
            history_stage2,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
        )
        _write_training_manifest(
            run_mode="finalized_from_best_checkpoint",
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            stage1_configured_epochs=args.stage1_epochs,
            stage2_configured_epochs=args.stage2_epochs,
            stage1_completed_epochs=len(history_stage1["loss"]),
            stage2_completed_epochs=len(history_stage2["loss"]),
            training_seconds=None,
            completion_note=(
                "Training was stopped after the target had already been exceeded. "
                "The stage-2 best-validation checkpoint is used as the final model."
            ),
        )
        metrics = _run_and_save_benchmark(
            model,
            (x_test, y_test),
            args,
            run_start=run_start,
            training_seconds=None,
            run_mode="finalized_from_best_checkpoint",
        )
        if not metrics["accuracy_requirement_met"]:
            raise SystemExit(
                f"Test accuracy {metrics['accuracy']:.4f} did not exceed "
                f"the required {ACCURACY_TARGET:.2f}."
            )
        return

    train_dataset, validation_dataset, _ = make_datasets(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
    )

    # 创建并训练模型
    model = create_baseline_model()
    _save_model_summary(model)

    # 使用微调策略进行训练
    # 1. 仅训练分类头（冻结基础模型）
    # 2. 解冻基础模型，并使用较低学习率进行微调
    training_start = time.perf_counter()
    model, history_stage1, history_stage2 = train_baseline_model(
        model,
        train_dataset,
        validation_dataset,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )
    training_seconds = time.perf_counter() - training_start
    _save_training_artifacts(
        history_stage1,
        history_stage2,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )
    _write_training_manifest(
        run_mode=(
            "train_and_benchmark" if not args.no_benchmark else "train_only"
        ),
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        stage1_configured_epochs=args.stage1_epochs,
        stage2_configured_epochs=args.stage2_epochs,
        stage1_completed_epochs=len(history_stage1.history["loss"]),
        stage2_completed_epochs=len(history_stage2.history["loss"]),
        training_seconds=training_seconds,
        completion_note="Training and benchmark completed normally.",
    )

    # 保存训练好的模型
    model.save(MODEL_PATH)

    # 进行基准测试
    if args.no_benchmark:
        print(f"Model saved to: {MODEL_PATH}")
        print("Benchmark skipped because --no-benchmark was supplied.")
        return

    metrics = _run_and_save_benchmark(
        model,
        (x_test, y_test),
        args,
        run_start=run_start,
        training_seconds=training_seconds,
        run_mode="train_and_benchmark",
    )

    if not metrics["accuracy_requirement_met"]:
        raise SystemExit(
            f"Test accuracy {metrics['accuracy']:.4f} did not exceed "
            f"the required {ACCURACY_TARGET:.2f}."
        )


if __name__ == "__main__":
    main()
