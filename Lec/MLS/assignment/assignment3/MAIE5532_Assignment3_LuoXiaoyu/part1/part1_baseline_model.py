#!/usr/bin/env python3
"""Train and benchmark the MobileNetV2 baseline for CIFAR-10."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
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
TRAIN_BATCH_SIZE = 64
VALIDATION_FRACTION = 0.1
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 14
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-4
ACCURACY_TARGET = 0.85

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "part1_results"
MODEL_PATH = OUTPUT_DIR / "baseline_mobilenetv2.keras"


def configure_runtime() -> None:
    """Set the random seed and permit incremental GPU memory allocation."""
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
    """Create the ImageNet-pretrained MobileNetV2 classification model."""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_preprocessing",
    )(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.2, name="classifier_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(inputs, outputs, name="cifar10_mobilenetv2_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data():
    """Load CIFAR-10 and retain pixels in the 0-255 range."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (
        x_train.astype(np.float32),
        y_train.reshape(-1).astype(np.int64),
        x_test.astype(np.float32),
        y_test.reshape(-1).astype(np.int64),
    )


def _stratified_split(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training data into stratified training and validation sets."""
    rng = np.random.default_rng(SEED)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        validation_size = max(
            1,
            int(round(len(class_indices) * VALIDATION_FRACTION)),
        )
        validation_parts.append(class_indices[:validation_size])
        train_parts.append(class_indices[validation_size:])

    return (
        rng.permutation(np.concatenate(train_parts)),
        rng.permutation(np.concatenate(validation_parts)),
    )


def _evaluation_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    """Create a resize-only evaluation dataset."""

    def prepare(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        return image, label

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create the augmented training pipeline and evaluation pipelines."""
    train_indices, validation_indices = _stratified_split(y_train)
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.03, seed=SEED + 1),
            layers.RandomZoom(0.08, 0.08, seed=SEED + 2),
            layers.RandomTranslation(0.05, 0.05, seed=SEED + 3),
        ]
    )

    def prepare_train(image: tf.Tensor, label: tf.Tensor):
        image = tf.cast(image, tf.float32)
        image = augmentation(image, training=True)
        image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
        return image, label

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (x_train[train_indices], y_train[train_indices])
        )
        .shuffle(len(train_indices), seed=SEED, reshuffle_each_iteration=True)
        .map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(TRAIN_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = _evaluation_dataset(
        x_train[validation_indices],
        y_train[validation_indices],
        TRAIN_BATCH_SIZE,
    )
    test_dataset = _evaluation_dataset(
        x_test,
        y_test,
        TRAIN_BATCH_SIZE,
    )
    return train_dataset, validation_dataset, test_dataset


def _write_history(
    path: Path,
    history: keras.callbacks.History,
) -> None:
    """Write one stage's epoch-level training history as CSV."""
    keys = list(history.history)
    rows = [
        ",".join(["epoch", *keys]),
        *[
            ",".join([str(epoch), *[str(history.history[key][epoch]) for key in keys]])
            for epoch in range(len(history.history["loss"]))
        ],
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train_baseline_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """Train the classification head, then fine-tune MobileNetV2."""
    base_model = model.get_layer("mobilenetv2_base")
    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history_stage1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=STAGE1_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=STAGE1_EPOCHS,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    _write_history(RESULTS_DIR / "training_stage1_log.csv", history_stage1)

    base_model.trainable = True
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
        epochs=STAGE2_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=STAGE2_EPOCHS,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    _write_history(RESULTS_DIR / "training_stage2_log.csv", history_stage2)
    return model


def _model_weight_memory_bytes(model: keras.Model) -> int:
    return sum(
        int(np.prod(weight.shape)) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )


def _estimate_flops(model: keras.Model) -> int:
    """Estimate float operations for one image."""
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


def benchmark_baseline_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    inference_runs: int = 100,
    batch_inference_runs: int = 40,
    warmup_runs: int = 20,
) -> dict[str, object]:
    """Evaluate accuracy and measure single-image and batch inference."""
    evaluation = model.evaluate(test_dataset, verbose=0, return_dict=True)
    images = tf.concat([batch for batch, _ in test_dataset.take(2)], axis=0)
    single_images = tf.image.resize(
        images[:inference_runs],
        IMAGE_SIZE,
        method="bilinear",
    )
    batch_images = tf.image.resize(
        images[:batch_size],
        IMAGE_SIZE,
        method="bilinear",
    )

    @tf.function(input_signature=[tf.TensorSpec((1, *INPUT_SHAPE), tf.float32)])
    def predict_one(inputs):
        return model(inputs, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec((batch_size, *INPUT_SHAPE), tf.float32)]
    )
    def predict_batch(inputs):
        return model(inputs, training=False)

    for _ in range(warmup_runs):
        predict_one(single_images[:1]).numpy()
        predict_batch(batch_images).numpy()

    single_times: list[float] = []
    for index in range(inference_runs):
        sample = single_images[index:index + 1]
        start = time.perf_counter()
        predict_one(sample).numpy()
        single_times.append((time.perf_counter() - start) * 1000.0)

    batch_times: list[float] = []
    for _ in range(batch_inference_runs):
        start = time.perf_counter()
        predict_batch(batch_images).numpy()
        batch_times.append((time.perf_counter() - start) * 1000.0)

    return {
        "single_inference_time_ms": float(np.mean(single_times)),
        "batch_inference_time_ms": float(np.mean(batch_times)),
        "memory_usage_mb": _model_weight_memory_bytes(model) / (1024 ** 2),
        "memory_usage_scope": (
            "TensorFlow model weights, including BatchNorm state; "
            "excludes activations and runtime allocators."
        ),
        "model_size_mb": MODEL_PATH.stat().st_size / (1024 ** 2),
        "accuracy": float(evaluation["accuracy"]),
        "flops": _estimate_flops(model),
        "parameters": int(model.count_params()),
        "energy_j_per_inference": None,
        "energy_status": (
            "Not measured: no reliable per-inference energy counter was "
            "available for the WSL GPU process."
        ),
    }


def main() -> None:
    """Run the complete Part 1 training and benchmark workflow."""
    configure_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    train_dataset, validation_dataset, test_dataset = _make_datasets(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    model = create_baseline_model()
    model = train_baseline_model(
        model,
        train_dataset,
        validation_dataset,
    )
    model.save(MODEL_PATH)

    metrics = benchmark_baseline_model(model, test_dataset)
    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))

    if metrics["accuracy"] <= ACCURACY_TARGET:
        raise SystemExit(
            f"Test accuracy {metrics['accuracy']:.4f} did not exceed "
            f"the required {ACCURACY_TARGET:.2f}."
        )


if __name__ == "__main__":
    main()
