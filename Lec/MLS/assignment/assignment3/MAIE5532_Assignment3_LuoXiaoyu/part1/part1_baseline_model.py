#!/usr/bin/env python3
"""第 1 部分：MobileNetV2 图像分类基线模型。"""

from __future__ import annotations

import json
import os
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
BATCH_SIZE = 64
VALIDATION_FRACTION = 0.1
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 25
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-4
ACCURACY_TARGET = 0.85

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "part1_results"
MODEL_PATH = OUTPUT_DIR / "baseline_mobilenetv2.keras"


def configure_runtime() -> None:
    """设置随机种子，并允许 GPU 按需分配显存。"""
    keras.utils.set_random_seed(SEED)
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
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

    # 将图像缩放到 224x224，以适配 MobileNetV2
    # 归一化像素值，并应用数据增强
    # 使用 tf.data.Dataset 高效加载数据
    # 保留 0-255 的像素范围，由模型内的 Rescaling 层统一归一化。
    return (
        x_train.astype(np.float32),
        y_train.reshape(-1).astype(np.int64),
        x_test.astype(np.float32),
        y_test.reshape(-1).astype(np.int64),
    )


def _stratified_split(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按类别划分训练集和验证集，测试集不参与训练。"""
    rng = np.random.default_rng(SEED)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        validation_count = max(
            1,
            int(round(len(class_indices) * VALIDATION_FRACTION)),
        )
        validation_indices.append(class_indices[:validation_count])
        train_indices.append(class_indices[validation_count:])

    return (
        rng.permutation(np.concatenate(train_indices)),
        rng.permutation(np.concatenate(validation_indices)),
    )


def _resize_images(images: tf.Tensor) -> tf.Tensor:
    """将图像缩放到 MobileNetV2 所需尺寸。"""
    return tf.image.resize(
        tf.cast(images, tf.float32),
        IMAGE_SIZE,
        method="bilinear",
    )


def _make_train_dataset(
    images: np.ndarray,
    labels: np.ndarray,
) -> tf.data.Dataset:
    """创建训练集流水线，并应用数据增强。"""
    augmentation = keras.Sequential(
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
        ]
    )

    def prepare(image: tf.Tensor, label: tf.Tensor):
        image = augmentation(image, training=True)
        return _resize_images(image), label

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
        .map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def _make_eval_dataset(
    images: np.ndarray,
    labels: np.ndarray,
) -> tf.data.Dataset:
    """创建不打乱、不增强的验证集或测试集流水线。"""

    def prepare(image: tf.Tensor, label: tf.Tensor):
        return _resize_images(image), label

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """划分训练集，并创建训练和验证流水线。"""
    train_indices, validation_indices = _stratified_split(y_train)
    return (
        _make_train_dataset(
            x_train[train_indices],
            y_train[train_indices],
        ),
        _make_eval_dataset(
            x_train[validation_indices],
            y_train[validation_indices],
        ),
    )


def _get_mobilenet_backbone(model: keras.Model) -> keras.Model:
    """从外层模型中定位 MobileNetV2 基础模型。"""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2") and hasattr(layer, "layers"):
            return layer
    raise ValueError("无法在模型中定位 MobileNetV2 基础模型。")


def _stage_callbacks(
    stage_name: str,
    epochs: int,
) -> list[keras.callbacks.Callback]:
    """保存每阶段的训练日志，并恢复验证准确率最佳权重。"""
    return [
        keras.callbacks.CSVLogger(
            str(RESULTS_DIR / f"training_{stage_name}_log.csv"),
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=epochs,
            restore_best_weights=True,
        ),
    ]


def train_baseline_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """先训练分类头，再以较低学习率微调 MobileNetV2。"""
    # 使用微调策略进行训练
    # 1. 仅训练分类头（冻结基础模型）
    base_model = _get_mobilenet_backbone(model)
    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=STAGE1_EPOCHS,
        callbacks=_stage_callbacks("stage1", STAGE1_EPOCHS),
        verbose=2,
    )

    # 2. 解冻基础模型，并使用较低学习率进行微调
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=STAGE2_EPOCHS,
        callbacks=_stage_callbacks("stage2", STAGE2_EPOCHS),
        verbose=2,
    )
    return model


def _model_weight_memory_bytes(model: keras.Model) -> int:
    """计算模型权重和 BatchNorm 状态的存储字节数。"""
    return sum(
        int(np.prod(weight.shape)) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )


def _estimate_flops(model: keras.Model) -> int:
    """估算单张图像前向计算所需的浮点运算次数。"""
    concrete_function = tf.function(
        lambda images: model(images, training=False)
    ).get_concrete_function(tf.TensorSpec((1, *INPUT_SHAPE), tf.float32))
    frozen_function = convert_variables_to_constants_v2(concrete_function)
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options["output"] = "none"
    profile = tf.compat.v1.profiler.profile(
        graph=frozen_function.graph,
        run_meta=tf.compat.v1.RunMetadata(),
        cmd="op",
        options=options,
    )
    return int(profile.total_float_ops)


def _timed_inference(
    model: keras.Model,
    inputs: tf.Tensor,
    runs: int,
) -> float:
    """测量固定输入形状下前向推理的平均延迟，单位为毫秒。"""
    batch_size = int(inputs.shape[0])

    @tf.function(
        input_signature=[
            tf.TensorSpec((batch_size, *INPUT_SHAPE), tf.float32)
        ]
    )
    def infer(images):
        return model(images, training=False)

    for _ in range(20):
        infer(inputs).numpy()

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        infer(inputs).numpy()
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings_ms))


def benchmark_baseline_model(
    model: keras.Model,
    test_data: tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
) -> dict[str, float | int]:
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
    test_dataset = _make_eval_dataset(x_test, y_test)
    evaluation = model.evaluate(
        test_dataset,
        verbose=0,
        return_dict=True,
    )

    # 测量单样本和批量推理延迟
    single_image = _resize_images(x_test[:1])
    batch_images = _resize_images(x_test[:batch_size])
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    single_inference_time = _timed_inference(
        model,
        single_image,
        runs=100,
    )
    batch_inference_time = _timed_inference(
        model,
        batch_images,
        runs=40,
    )
    rss_after = process.memory_info().rss
    runtime_memory_mb = max(
        0.0,
        (rss_after - rss_before) / (1024 ** 2),
    )
    weight_memory_mb = _model_weight_memory_bytes(model) / (1024 ** 2)
    metrics = {
        'single_inference_time': single_inference_time,
        'batch_inference_time': batch_inference_time,
        'memory_usage_mb': max(runtime_memory_mb, weight_memory_mb),
        'model_size_mb': MODEL_PATH.stat().st_size / (1024 ** 2),
        'accuracy': float(evaluation["accuracy"]),
        'flops': _estimate_flops(model),
        'parameters': int(model.count_params()),
    }
    return metrics


def main() -> None:
    """执行 Part 1 的数据加载、训练、保存和基准测试。"""
    configure_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    train_dataset, validation_dataset = make_datasets(
        x_train,
        y_train,
    )

    # 创建并训练模型
    model = create_baseline_model()
    model = train_baseline_model(
        model,
        train_dataset,
        validation_dataset,
    )

    # 保存训练好的模型
    model.save(MODEL_PATH)

    # 进行基准测试
    metrics = benchmark_baseline_model(model, (x_test, y_test))
    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if metrics["accuracy"] <= ACCURACY_TARGET:
        raise SystemExit(
            f"Test accuracy {metrics['accuracy']:.4f} did not exceed "
            f"the required {ACCURACY_TARGET:.2f}."
        )


if __name__ == "__main__":
    main()
