#!/usr/bin/env python3
"""MobileNetV2 image classification baseline."""

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
    """Set the random seed and allow on-demand GPU memory allocation."""
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
    Create a MobileNetV2-based model for image classification.

    Args:
        input_shape: Input image dimensions
        num_classes: Number of classification classes

    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Base model: MobileNetV2 (pretrained on ImageNet, frozen initially)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_input")
    # MobileNetV2 ImageNet weights expect inputs in the range [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_preprocessing",
    )(inputs)
    x = base_model(x, training=False)

    # Add a custom classification head for the dataset
    # Include global average pooling and dropout for regularization
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
    Load CIFAR-10 dataset and preprocess for MobileNetV2.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Resize images to 224x224 for MobileNetV2
    # Normalize pixel values and apply data augmentation
    # Use tf.data.Dataset for efficient data loading
    # Keep pixels in [0, 255]; the model applies ImageNet normalization.
    return (
        x_train.astype(np.float32),
        y_train.reshape(-1).astype(np.int64),
        x_test.astype(np.float32),
        y_test.reshape(-1).astype(np.int64),
    )


def _stratified_split(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a class-stratified training and validation split."""
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
    """Resize images to the dimensions expected by MobileNetV2."""
    return tf.image.resize(
        tf.cast(images, tf.float32),
        IMAGE_SIZE,
        method="bilinear",
    )


def _make_train_dataset(
    images: np.ndarray,
    labels: np.ndarray,
) -> tf.data.Dataset:
    """Create the augmented training dataset."""
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
    """Create a deterministic dataset without augmentation."""

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
    """Split the training data and build the train/validation pipelines."""
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
    """Locate the MobileNetV2 backbone in the outer model."""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2") and hasattr(layer, "layers"):
            return layer
    raise ValueError("Could not locate the MobileNetV2 backbone in the model.")


def _stage_callbacks(
    stage_name: str,
    epochs: int,
) -> list[keras.callbacks.Callback]:
    """Log each training stage and restore its best validation weights."""
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
    """Train the classification head, then fine-tune MobileNetV2."""
    # Use a two-stage fine-tuning strategy
    # 1. Train the classification head only (base model frozen)
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

    # 2. Unfreeze the base model and fine-tune with a lower learning rate
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
    """Calculate storage used by model weights and BatchNorm state."""
    return sum(
        int(np.prod(weight.shape)) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )


def _estimate_flops(model: keras.Model) -> int:
    """Estimate floating-point operations for one forward pass."""
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
    """Measure mean forward-pass latency in milliseconds for fixed input."""
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
    Benchmark the baseline model for latency and memory usage.

    Args:
        model: Trained Keras model
        test_data: Test dataset
        batch_size: Batch size for inference

    Returns:
        dict: Performance metrics
    """
    x_test, y_test = test_data
    test_dataset = _make_eval_dataset(x_test, y_test)
    evaluation = model.evaluate(
        test_dataset,
        verbose=0,
        return_dict=True,
    )

    # Measure inference latency for a single sample and a batch
    # Monitor memory usage during inference
    # Calculate FLOPs and model parameters
    # Energy is not measured because no reliable per-inference counter is available.
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
    """Load data, train, save, and benchmark the baseline model."""
    configure_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    train_dataset, validation_dataset = make_datasets(
        x_train,
        y_train,
    )

    # Create and train the model
    model = create_baseline_model()
    model = train_baseline_model(
        model,
        train_dataset,
        validation_dataset,
    )

    # Save the trained model
    model.save(MODEL_PATH)

    # Benchmark the trained model
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
            f"the configured target of {ACCURACY_TARGET:.2f}."
        )


if __name__ == "__main__":
    main()
