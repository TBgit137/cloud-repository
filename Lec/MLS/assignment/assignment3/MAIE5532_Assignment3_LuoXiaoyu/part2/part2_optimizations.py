#!/usr/bin/env python3
"""MobileNetV2 hardware-aware optimization."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# XLA can misresolve the NVIDIA libdevice path in this environment.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
BATCH_SIZE = 64
VAL_SPLIT = 0.1
HEAD_EPOCHS = 1
FINETUNE_EPOCHS = 3
DISTILL_EPOCHS = 3
PRUNE_EPOCHS = 2
PRUNE_SPARSITY = 0.3
QAT_EPOCHS = 1
REPRESENTATIVE_SAMPLES = 200
EARLY_EXIT_THRESHOLD = 0.7
ACCURACY_TARGET = 0.85

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART1_DIR = ROOT / "part1"
BASELINE_PATH = PART1_DIR / "baseline_mobilenetv2.keras"
BASELINE_METRICS_PATH = PART1_DIR / "part1_results" / "baseline_metrics.json"
OUTPUT_DIR = HERE / "part2_results"
MODEL_DIR = OUTPUT_DIR / "optimized_models"
QUANTIZED_DIR = OUTPUT_DIR / "quantized_models"
LATENCY_MODEL_PATH = MODEL_DIR / "latency_optimized.keras"
MEMORY_MODEL_PATH = MODEL_DIR / "memory_optimized.keras"
ENERGY_MODEL_PATH = MODEL_DIR / "energy_optimized.keras"

sys.path.insert(0, str(PART1_DIR))
from part1_baseline_model import load_and_preprocess_data  # noqa: E402


def configure_runtime() -> None:
    """Set the random seed and allow on-demand GPU memory allocation."""
    keras.utils.set_random_seed(SEED)
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass


def _build_classifier(
    input_size: int,
    alpha: float,
    name: str,
) -> keras.Model:
    """Create a classifier with ImageNet-pretrained MobileNetV2."""
    input_shape = (input_size, input_size, 3)
    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name=f"{name}_preprocessing",
    )(inputs)
    base_model = keras.applications.MobileNetV2(
        input_tensor=x,
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name=f"{name}_pooling")(x)
    x = layers.Dropout(0.2, name=f"{name}_dropout")(x)
    outputs = layers.Dense(
        10,
        activation="softmax",
        name=f"{name}_classifier",
    )(x)
    model = keras.Model(inputs, outputs, name=name)
    return compile_model(model)


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Compile the model with a common loss and accuracy metric."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


def _head_layers(model: keras.Model) -> list[layers.Layer]:
    """Return custom head layers so they can be separated from the backbone."""
    selected = []
    for layer in model.layers:
        name = layer.name.lower()
        if (
            "pooling" in name
            or "dropout" in name
            or "classifier" in name
        ):
            selected.append(layer)
    if not selected:
        raise ValueError("Could not locate the classification head in the model.")
    return selected


def _get_mobilenet_backbone(model: keras.Model) -> keras.Model:
    """Locate the MobileNetV2 backbone in the outer model."""
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower() and hasattr(layer, "layers"):
            return layer
    raise ValueError("Could not locate the MobileNetV2 backbone in the model.")


def _walk_layers(model: keras.Model):
    """Yield all layers in a potentially nested model."""
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from _walk_layers(layer)


def _set_head_only_training(model: keras.Model) -> None:
    """Freeze the MobileNetV2 backbone and train only the custom head."""
    head_ids = {id(layer) for layer in _head_layers(model)}
    for layer in _walk_layers(model):
        layer.trainable = id(layer) in head_ids


def _unfreeze_mobilenet(model: keras.Model) -> None:
    """Unfreeze backbone convolutions while keeping BatchNorm and preprocessing frozen."""
    model.trainable = True
    for layer in _walk_layers(model):
        layer.trainable = True
    for layer in _walk_layers(model):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        if layer.name.startswith("tf.") or "preprocessing" in layer.name:
            layer.trainable = False


def create_latency_optimized_model() -> keras.Model:
    """Create a model optimized for inference latency."""
    # Use MobileNetV2 with alpha=0.5 and reduce the input to 160x160.
    # Depthwise-separable convolution is efficient for mobile inference.
    return _build_classifier(
        input_size=160,
        alpha=0.5,
        name="latency_optimized",
    )


def create_memory_optimized_model() -> keras.Model:
    """Create a model optimized for memory footprint."""
    # Reduce width and input resolution to compress the structured model.
    # Knowledge distillation and magnitude pruning are applied during training.
    return _build_classifier(
        input_size=160,
        alpha=0.5,
        name="memory_optimized",
    )


def create_energy_optimized_model() -> keras.Model:
    """Create an energy-oriented model with an early-exit branch."""
    input_shape = (160, 160, 3)
    inputs = keras.Input(shape=input_shape, name="energy_optimized_input")
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="energy_optimized_preprocessing",
    )(inputs)
    base_model = keras.applications.MobileNetV2(
        input_tensor=x,
        input_shape=input_shape,
        alpha=0.5,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    features = base_model.output

    # Attach an early classifier so confident samples can skip the full backbone.
    early_features = base_model.get_layer("block_6_project").output
    early_exit = layers.GlobalAveragePooling2D(
        name="early_exit_pooling"
    )(early_features)
    early_exit = layers.Dense(
        10,
        activation="softmax",
        name="early_exit_classifier",
    )(early_exit)

    # ReLU6 and a simple linear classifier avoid unnecessary computation.
    final_features = layers.GlobalAveragePooling2D(
        name="energy_optimized_pooling"
    )(features)
    final_features = layers.Dropout(
        0.2,
        name="energy_optimized_dropout",
    )(final_features)
    final_output = layers.Dense(
        10,
        activation="softmax",
        name="energy_optimized_classifier",
    )(final_features)

    model = keras.Model(
        inputs,
        {
            "early_exit": early_exit,
            "classifier": final_output,
        },
        name="energy_optimized",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "early_exit": keras.losses.SparseCategoricalCrossentropy(),
            "classifier": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"early_exit": 0.3, "classifier": 1.0},
        metrics={
            "early_exit": ["accuracy"],
            "classifier": ["accuracy"],
        },
    )
    return model


def create_optimized_models() -> dict[str, keras.Model]:
    """
    Create hardware-optimized variants of the baseline model.

    Returns:
        dict: Models optimized for different hardware constraints
    """
    models = {}

    # Build the latency-optimized model
    models["latency_optimized"] = create_latency_optimized_model()

    # Build the memory-optimized model
    models["memory_optimized"] = create_memory_optimized_model()

    # Build the energy-oriented model
    models["energy_optimized"] = create_energy_optimized_model()
    return models


def _stratified_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create a class-stratified training and validation split."""
    rng = np.random.default_rng(SEED)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        val_count = max(1, int(round(len(indices) * VAL_SPLIT)))
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])
    return (
        rng.permutation(np.concatenate(train_parts)),
        rng.permutation(np.concatenate(val_parts)),
    )


def _augmentation() -> keras.Sequential:
    """Create the training-time augmentation pipeline."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.03, fill_mode="nearest", seed=SEED + 1),
            layers.RandomZoom(0.08, 0.08, fill_mode="nearest", seed=SEED + 2),
            layers.RandomTranslation(
                0.05,
                0.05,
                fill_mode="nearest",
                seed=SEED + 3,
            ),
        ]
    )


def _resize(images: tf.Tensor, input_size: int) -> tf.Tensor:
    """Resize images to the target model input size."""
    return tf.image.resize(
        tf.cast(images, tf.float32),
        (input_size, input_size),
        method="bilinear",
    )


def _classification_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    input_size: int,
    *,
    training: bool,
) -> tf.data.Dataset:
    """Create a classification dataset for the requested input size."""
    augmentation = _augmentation()

    def prepare(image: tf.Tensor, label: tf.Tensor):
        if training:
            image = augmentation(image, training=True)
        return _resize(image, input_size), label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            len(images),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return (
        dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def _energy_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    training: bool,
) -> tf.data.Dataset:
    """Provide the same labels to the full model and early-exit branch."""
    dataset = _classification_dataset(
        images,
        labels,
        input_size=160,
        training=training,
    )

    def duplicate_labels(model_inputs, model_labels):
        return model_inputs, {
            "early_exit": model_labels,
            "classifier": model_labels,
        }

    return dataset.map(duplicate_labels)


def _compile_energy_model(model: keras.Model, learning_rate: float) -> None:
    """Compile the energy model with separate losses for both outputs."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "early_exit": keras.losses.SparseCategoricalCrossentropy(),
            "classifier": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"early_exit": 0.3, "classifier": 1.0},
        metrics={
            "early_exit": ["accuracy"],
            "classifier": ["accuracy"],
        },
        jit_compile=False,
    )


def _train_classifier(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """Train the head with a frozen backbone, then fine-tune at a lower rate."""
    _set_head_only_training(model)
    compile_model(model, learning_rate=1e-3)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=HEAD_EPOCHS,
        verbose=2,
    )

    _unfreeze_mobilenet(model)
    compile_model(model, learning_rate=1e-4)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINETUNE_EPOCHS,
        verbose=2,
    )
    return model


def _train_energy_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """Train the early-exit and full classification branches."""
    _set_head_only_training(model)
    _compile_energy_model(model, learning_rate=1e-3)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=HEAD_EPOCHS,
        verbose=2,
    )

    _unfreeze_mobilenet(model)
    _compile_energy_model(model, learning_rate=1e-4)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINETUNE_EPOCHS,
        verbose=2,
    )
    return model


def _dual_input_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    student_size: int,
    teacher_size: int,
) -> tf.data.Dataset:
    """Create a dataset that supplies student and teacher inputs."""
    augmentation = _augmentation()

    def prepare(image: tf.Tensor, label: tf.Tensor):
        image = augmentation(image, training=True)
        return (
            _resize(image, student_size),
            _resize(image, teacher_size),
        ), label

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
        .map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def _train_with_distillation(
    student: keras.Model,
    teacher: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """Train the compact student model using the frozen baseline teacher."""
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(student_images, teacher_images, labels):
        teacher_probabilities = teacher(teacher_images, training=False)
        with tf.GradientTape() as tape:
            student_probabilities = student(student_images, training=True)
            hard_loss = keras.losses.sparse_categorical_crossentropy(
                labels,
                student_probabilities,
            )
            soft_loss = keras.losses.KLDivergence()(
                teacher_probabilities,
                student_probabilities,
            )
            loss = tf.reduce_mean(0.5 * hard_loss + 0.5 * soft_loss)
        gradients = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    for epoch in range(DISTILL_EPOCHS):
        losses: list[float] = []
        for (student_images, teacher_images), labels in train_dataset:
            losses.append(float(train_step(
                student_images,
                teacher_images,
                labels,
            )))
        print(
            f"Distillation epoch {epoch + 1}/{DISTILL_EPOCHS}: "
            f"loss={np.mean(losses):.4f}"
        )
    _unfreeze_mobilenet(student)
    compile_model(student, learning_rate=1e-4)
    student.fit(
        train_dataset.map(lambda inputs, labels: (inputs[0], labels)),
        validation_data=validation_dataset,
        epochs=FINETUNE_EPOCHS,
        verbose=2,
    )
    return student


def _apply_unstructured_pruning(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.Model:
    """Prune weights by magnitude and recover with a fixed mask."""
    masks = []
    for variable in model.weights:
        array = variable.numpy()
        if array.ndim < 2 or "kernel" not in variable.name:
            continue
        threshold = np.quantile(
            np.abs(array),
            PRUNE_SPARSITY,
        )
        mask = np.abs(array) > threshold
        variable.assign((array * mask).astype(array.dtype))
        masks.append((variable, mask))

    compile_model(model, learning_rate=1e-5)
    optimizer = model.optimizer
    for epoch in range(PRUNE_EPOCHS):
        losses: list[float] = []
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(
                    labels,
                    predictions,
                )
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            for variable, mask in masks:
                variable.assign(variable.numpy() * mask)
            losses.append(float(loss))
        validation = model.evaluate(validation_dataset, verbose=0)
        print(
            f"Pruning epoch {epoch + 1}/{PRUNE_EPOCHS}: "
            f"loss={np.mean(losses):.4f}, validation={validation}"
        )
    return compile_model(model, learning_rate=1e-4)


def representative_dataset_generator(x_train_sample):
    """Yield representative training samples for quantization calibration."""
    for image in x_train_sample:
        yield [np.expand_dims(image.astype(np.float32), axis=0)]


def _convert_tflite(
    model: keras.Model,
    output_path: Path,
    mode: str,
    representative_data,
) -> Path:
    """Convert a Keras model to TFLite with the requested precision."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "mixed_precision":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(f"Unsupported TFLite quantization mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())
    return output_path


def post_training_quantization(model, x_train_sample):
    """Apply INT8 post-training quantization with calibration samples."""
    input_size = int(model.input_shape[1])
    sample = np.stack(
        [
            tf.image.resize(
                image,
                (input_size, input_size),
                method="bilinear",
            ).numpy()
            for image in x_train_sample[:REPRESENTATIVE_SAMPLES]
        ]
    )
    path = _convert_tflite(
        model,
        QUANTIZED_DIR / "ptq_int8.tflite",
        mode="int8",
        representative_data=lambda: representative_dataset_generator(sample),
    )
    return {"path": str(path), "input_range": "0-255"}


def quantization_aware_training(model, x_train_sample):
    """Run quantization-aware training and export an INT8 TFLite model."""
    input_size = int(model.input_shape[1])
    sample = np.stack(
        [
            tf.image.resize(
                image,
                (input_size, input_size),
                method="bilinear",
            ).numpy()
            for image in x_train_sample[:REPRESENTATIVE_SAMPLES]
        ]
    )
    pseudo_labels = np.argmax(model(sample, training=False).numpy(), axis=1)
    qat_model = keras.models.clone_model(model)
    qat_model.set_weights(model.get_weights())
    compile_model(qat_model, learning_rate=1e-5)
    optimizer = qat_model.optimizer
    for epoch in range(QAT_EPOCHS):
        for start in range(0, len(sample), 32):
            batch_images = sample[start:start + 32]
            batch_labels = pseudo_labels[start:start + 32]
            with tf.GradientTape() as tape:
                predictions = qat_model(batch_images, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(
                    batch_labels,
                    predictions,
                )
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, qat_model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, qat_model.trainable_variables)
            )
            # Project convolution and Dense kernels onto an 8-bit weight grid.
            for variable in qat_model.trainable_variables:
                values = variable.numpy()
                if values.ndim < 2:
                    continue
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                if maximum <= minimum:
                    continue
                scale = (maximum - minimum) / 255.0
                quantized = minimum + np.round(
                    (values - minimum) / scale
                ) * scale
                variable.assign(quantized.astype(values.dtype))
        print(f"QAT epoch {epoch + 1}/{QAT_EPOCHS}: loss={float(loss):.4f}")
    path = _convert_tflite(
        qat_model,
        QUANTIZED_DIR / "qat_int8.tflite",
        mode="int8",
        representative_data=lambda: representative_dataset_generator(sample),
    )
    return {"path": str(path), "input_range": "0-255"}


def mixed_precision_quantization(model):
    """Export the model as mixed-precision FP16 TFLite."""
    path = _convert_tflite(
        model,
        QUANTIZED_DIR / "mixed_precision.tflite",
        mode="mixed_precision",
        representative_data=None,
    )
    return {"path": str(path), "input_range": "0-255"}


def dynamic_range_quantization(model):
    """Export the model with dynamic-range quantization."""
    path = _convert_tflite(
        model,
        QUANTIZED_DIR / "dynamic_range.tflite",
        mode="dynamic_range",
        representative_data=None,
    )
    return {"path": str(path), "input_range": "0-255"}


def apply_quantization_optimizations(model, x_train_sample):
    """
    Apply the supported quantization techniques to a trained model.

    Args:
        model: Trained Keras model
        x_train_sample: Representative training samples for calibration

    Returns:
        dict: Quantized model variants
    """
    quantized_models = {}
    quantized_models["ptq_int8"] = post_training_quantization(
        model,
        x_train_sample,
    )
    quantized_models["qat_int8"] = quantization_aware_training(
        model,
        x_train_sample,
    )
    quantized_models["mixed_precision"] = mixed_precision_quantization(model)
    quantized_models["dynamic_range"] = dynamic_range_quantization(model)
    return quantized_models


class GradientCheckpointedModel(keras.Model):
    """Use activation recomputation to reduce training memory."""

    def __init__(self, network: keras.Model):
        super().__init__()
        self.network = network

    def call(self, inputs, training=False):
        if training:
            return tf.recompute_grad(
                lambda values: self.network(values, training=True)
            )(inputs)
        return self.network(inputs, training=False)


class ShardedInferenceModel(keras.Model):
    """Place feature extraction and the classifier on separate devices."""

    def __init__(self, feature_model: keras.Model, head_model: keras.Model):
        super().__init__()
        self.feature_model = feature_model
        self.head_model = head_model

    def call(self, inputs, training=False):
        with tf.device("/GPU:0"):
            features = self.feature_model(inputs, training=False)
        with tf.device("/CPU:0"):
            return self.head_model(features, training=False)


class ActivationCompressedModel(keras.Model):
    """Use FP16 head activations to reduce inference activation memory."""

    def __init__(self, feature_model: keras.Model, head_model: keras.Model):
        super().__init__()
        self.feature_model = feature_model
        self.head_model = head_model

    def call(self, inputs, training=False):
        features = self.feature_model(inputs, training=False)
        features = tf.cast(features, tf.float16)
        with tf.device("/GPU:0"):
            return self.head_model(features, training=False)


def _split_feature_and_head(
    model: keras.Model,
    activation_dtype: str = "float32",
) -> tuple[keras.Model, keras.Model]:
    """Create weight-sharing feature extractor and head views of the model."""
    pooling_name = f"{model.name}_pooling"
    final_pooling = model.get_layer(pooling_name)
    feature_model = keras.Model(model.input, final_pooling.input)
    head_start = model.layers.index(final_pooling)
    head_layers = model.layers[head_start:]

    head_input = keras.Input(
        shape=feature_model.output_shape[1:],
        dtype=activation_dtype,
    )
    x = head_input
    for source_layer in head_layers:
        config = source_layer.get_config()
        config["dtype"] = activation_dtype
        x = source_layer.__class__.from_config(config)(x)
    head_model = keras.Model(head_input, x)
    head_model.set_weights([weight for layer in head_layers for weight in layer.get_weights()])
    return feature_model, head_model


def implement_gradient_checkpointing(model: keras.Model):
    """Wrap a model's training forward pass with gradient checkpointing."""
    return GradientCheckpointedModel(model)


def implement_model_sharding(model: keras.Model):
    """Split feature extraction and classification into separate devices."""
    feature_model, head_model = _split_feature_and_head(model)
    return ShardedInferenceModel(feature_model, head_model)


def implement_activation_compression(model: keras.Model):
    """Compress classification-head activations using FP16."""
    feature_model, head_model = _split_feature_and_head(
        model,
        activation_dtype="float16",
    )
    return ActivationCompressedModel(feature_model, head_model)


def find_optimal_batch_size(model: keras.Model):
    """Select the highest-throughput batch size from common candidates."""
    input_size = int(model.input_shape[1])
    candidates = (1, 8, 16, 32, 64)
    results = []
    for batch_size in candidates:
        inputs = tf.zeros((batch_size, input_size, input_size, 3))

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    (batch_size, input_size, input_size, 3),
                    tf.float32,
                )
            ]
        )
        def infer_batch(images):
            return model(images, training=False)

        for _ in range(3):
            infer_batch(inputs).numpy()
        start = time.perf_counter()
        infer_batch(inputs).numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append({
            "batch_size": batch_size,
            "latency_ms": elapsed_ms,
            "throughput_samples_per_second": batch_size * 1000.0 / elapsed_ms,
        })
    best = max(results, key=lambda item: item["throughput_samples_per_second"])
    return {"candidates": results, "best": best}


def implement_memory_optimizations(model):
    """
    Apply the supported memory-aware optimization techniques.

    Args:
        model: Base model to optimize

    Returns:
        dict: Memory optimization results
    """
    optimizations = {}
    optimizations["gradient_checkpointing"] = implement_gradient_checkpointing(model)
    optimizations["model_sharding"] = implement_model_sharding(model)
    optimizations["activation_compression"] = implement_activation_compression(model)
    optimizations["optimal_batch_size"] = find_optimal_batch_size(model)
    return optimizations


def early_exit_predictions(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    threshold: float = EARLY_EXIT_THRESHOLD,
) -> dict[str, float | int]:
    """Route confident samples to the early classifier and others to the full model."""
    early_model = keras.Model(model.input, model.output["early_exit"])
    full_model = keras.Model(model.input, model.output["classifier"])
    input_size = int(model.input_shape[1])
    dataset = _classification_dataset(
        images,
        np.zeros(len(images), dtype=np.int64),
        input_size=input_size,
        training=False,
    )
    correct = 0
    total = 0
    early_count = 0
    for image_batch, _ in dataset:
        early_probabilities = early_model(image_batch, training=False).numpy()
        confidence = np.max(early_probabilities, axis=1)
        confident = confidence >= threshold
        predictions = np.argmax(early_probabilities, axis=1)
        if np.any(~confident):
            full_batch = tf.gather(
                image_batch,
                np.flatnonzero(~confident),
            )
            full_predictions = np.argmax(
                full_model(full_batch, training=False).numpy(),
                axis=1,
            )
            predictions[~confident] = full_predictions
        early_count += int(confident.sum())
        total += len(image_batch)
        start = total - len(image_batch)
        correct += int(np.sum(predictions == labels[start:total]))
    return {
        "samples": total,
        "early_exit_samples": early_count,
        "early_exit_fraction": early_count / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def _predict_from_model(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Return class predictions from a single-output model."""
    input_size = int(model.input_shape[1])
    dataset = _classification_dataset(
        images,
        np.zeros(len(images), dtype=np.int64),
        input_size=input_size,
        training=False,
    )
    predictions = []
    for image_batch, _ in dataset:
        outputs = model(image_batch, training=False)
        if isinstance(outputs, dict):
            outputs = outputs["classifier"]
        predictions.append(np.argmax(outputs.numpy(), axis=1))
    return np.concatenate(predictions)


def evaluate_keras_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate test accuracy and mean single-sample latency for a Keras model."""
    predictions = _predict_from_model(model, x_test)
    accuracy = float(np.mean(predictions == y_test))
    input_size = int(model.input_shape[1])
    sample = tf.convert_to_tensor(
        tf.image.resize(
            x_test[:1],
            (input_size, input_size),
            method="bilinear",
        )
    )

    @tf.function
    def infer(images):
        outputs = model(images, training=False)
        return outputs["classifier"] if isinstance(outputs, dict) else outputs

    for _ in range(20):
        infer(sample).numpy()
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        infer(sample).numpy()
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "accuracy": accuracy,
        "single_inference_time_ms": float(np.mean(timings)),
        "parameters": int(model.count_params()),
    }


def _latency_only(
    model: keras.Model,
    input_size: int,
    runs: int = 20,
) -> float:
    """Measure single-sample inference latency for a wrapped model."""
    sample = tf.zeros((1, input_size, input_size, 3))

    @tf.function(
        input_signature=[
            tf.TensorSpec((1, input_size, input_size, 3), tf.float32)
        ]
    )
    def synchronized_inference(images):
        outputs = model(images, training=False)
        if isinstance(outputs, dict):
            outputs = outputs["classifier"]
        return outputs

    for _ in range(5):
        synchronized_inference(sample).numpy()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        synchronized_inference(sample).numpy()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings))


def _weight_sparsity(model: keras.Model) -> float:
    """Calculate the fraction of zeros in convolution and Dense kernels."""
    total = 0
    zeros = 0
    for variable in model.weights:
        values = variable.numpy()
        if values.ndim < 2 or "kernel" not in variable.name:
            continue
        total += values.size
        zeros += int(np.count_nonzero(values == 0))
    return zeros / total if total else 0.0


def evaluate_tflite_model(
    path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    normalize_input: bool = False,
) -> dict[str, float]:
    """Evaluate test accuracy and mean latency for a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_size = int(input_detail["shape"][1])
    correct = 0
    for image, label in zip(x_test, y_test):
        model_input = tf.image.resize(
            image,
            (input_size, input_size),
            method="bilinear",
        ).numpy()[None, ...]
        if normalize_input:
            model_input = model_input / 127.5 - 1.0
        if input_detail["dtype"] in (np.int8, np.uint8):
            scale, zero_point = input_detail["quantization"]
            model_input = model_input / scale + zero_point
            model_input = np.clip(
                np.round(model_input),
                np.iinfo(input_detail["dtype"]).min,
                np.iinfo(input_detail["dtype"]).max,
            ).astype(input_detail["dtype"])
        else:
            model_input = model_input.astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_detail["index"])
        correct += int(np.argmax(prediction) == int(label))

    sample = tf.image.resize(
        x_test[:1],
        (input_size, input_size),
        method="bilinear",
    ).numpy()
    if normalize_input:
        sample = sample / 127.5 - 1.0
    if input_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = input_detail["quantization"]
        sample = np.clip(
            np.round(sample / scale + zero_point),
            np.iinfo(input_detail["dtype"]).min,
            np.iinfo(input_detail["dtype"]).max,
        ).astype(input_detail["dtype"])
    else:
        sample = sample.astype(input_detail["dtype"])
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "accuracy": correct / len(y_test),
        "single_inference_time_ms": float(np.mean(timings)),
        "model_size_mb": path.stat().st_size / (1024 ** 2),
    }


def main() -> None:
    """Train optimized models, export quantized variants, and save comparisons."""
    configure_runtime()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_PATH}")

    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    train_indices, val_indices = _stratified_split(y_train)
    baseline = keras.models.load_model(BASELINE_PATH, compile=False)

    optimized_models = create_optimized_models()
    latency_model = optimized_models["latency_optimized"]
    latency_train = _classification_dataset(
        x_train[train_indices],
        y_train[train_indices],
        input_size=160,
        training=True,
    )
    latency_val = _classification_dataset(
        x_train[val_indices],
        y_train[val_indices],
        input_size=160,
        training=False,
    )
    latency_model = _train_classifier(
        latency_model,
        latency_train,
        latency_val,
    )
    latency_model.save(MODEL_DIR / "latency_optimized.keras")

    memory_model = optimized_models["memory_optimized"]
    memory_train = _classification_dataset(
        x_train[train_indices],
        y_train[train_indices],
        input_size=160,
        training=True,
    )
    memory_val = _classification_dataset(
        x_train[val_indices],
        y_train[val_indices],
        input_size=160,
        training=False,
    )
    distillation_train = _dual_input_dataset(
        x_train[train_indices],
        y_train[train_indices],
        student_size=160,
        teacher_size=224,
    )
    memory_model = _train_with_distillation(
        memory_model,
        baseline,
        distillation_train,
        memory_val,
    )
    memory_model = _apply_unstructured_pruning(
        memory_model,
        memory_train,
        memory_val,
    )
    memory_model.save(MODEL_DIR / "memory_optimized.keras")

    energy_model = optimized_models["energy_optimized"]
    energy_train = _energy_dataset(
        x_train[train_indices],
        y_train[train_indices],
        training=True,
    )
    energy_val = _energy_dataset(
        x_train[val_indices],
        y_train[val_indices],
        training=False,
    )
    energy_model = _train_energy_model(
        energy_model,
        energy_train,
        energy_val,
    )
    energy_model.save(MODEL_DIR / "energy_optimized.keras")

    quantized_paths = apply_quantization_optimizations(
        latency_model,
        x_train[train_indices],
    )
    latency_path = MODEL_DIR / "latency_optimized.keras"
    memory_path = MODEL_DIR / "memory_optimized.keras"
    energy_path = MODEL_DIR / "energy_optimized.keras"
    memory_optimizations = implement_memory_optimizations(latency_model)
    memory_management = {
        "gradient_checkpointing": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["gradient_checkpointing"],
                input_size=160,
            ),
            "scope": "training activation-memory proxy",
        },
        "model_sharding": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["model_sharding"],
                input_size=160,
            ),
            "scope": "GPU feature extraction + CPU classification head",
        },
        "activation_compression": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["activation_compression"],
                input_size=160,
            ),
            "scope": "mixed_float16 classification-head activations",
        },
        "optimal_batch_size": memory_optimizations["optimal_batch_size"],
    }

    energy_metrics = evaluate_keras_model(energy_model, x_test, y_test)
    energy_metrics = {
        **energy_metrics,
        "full_model_accuracy": energy_metrics["accuracy"],
        **early_exit_predictions(energy_model, x_test, y_test),
        "path": str(energy_path),
        "model_size_mb": energy_path.stat().st_size / (1024 ** 2),
    }
    architecture_metrics = {
        "latency_optimized": {
            **evaluate_keras_model(latency_model, x_test, y_test),
            "path": str(latency_path),
            "model_size_mb": latency_path.stat().st_size / (1024 ** 2),
        },
        "memory_optimized": {
            **evaluate_keras_model(memory_model, x_test, y_test),
            "path": str(memory_path),
            "model_size_mb": memory_path.stat().st_size / (1024 ** 2),
            "kernel_sparsity": _weight_sparsity(memory_model),
        },
        "energy_optimized": energy_metrics,
    }
    baseline_accuracy = float(
        json.loads(
            BASELINE_METRICS_PATH.read_text(encoding="utf-8")
        )["accuracy"]
    )
    for metrics in architecture_metrics.values():
        metrics["accuracy_drop_vs_baseline"] = (
            baseline_accuracy - metrics["accuracy"]
        )
    quantized_metrics = {}
    for name, information in quantized_paths.items():
        path = Path(information["path"])
        quantized_metrics[name] = {
            **evaluate_tflite_model(
                path,
                x_test,
                y_test,
                normalize_input=information["input_range"] == "-1 to 1",
            ),
            "path": str(path),
            "input_range": information["input_range"],
            "source_model": "latency_optimized",
        }
        quantized_metrics[name]["accuracy_drop_vs_baseline"] = (
            baseline_accuracy - quantized_metrics[name]["accuracy"]
        )

    comparison = {
        "baseline": json.loads(
            BASELINE_METRICS_PATH.read_text(encoding="utf-8")
        ),
        "architecture_models": architecture_metrics,
        "quantized_models": quantized_metrics,
        "memory_management": memory_management,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "performance_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


def evaluate_existing_models() -> None:
    """Reuse trained models and rebuild the performance comparison."""
    configure_runtime()
    model_paths = {
        "latency_optimized": LATENCY_MODEL_PATH,
        "memory_optimized": MEMORY_MODEL_PATH,
        "energy_optimized": ENERGY_MODEL_PATH,
    }
    for path in model_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Trained model not found: {path}")

    latency_model = keras.models.load_model(
        LATENCY_MODEL_PATH,
        compile=False,
    )
    memory_model = keras.models.load_model(
        MEMORY_MODEL_PATH,
        compile=False,
    )
    energy_model = keras.models.load_model(
        ENERGY_MODEL_PATH,
        compile=False,
    )
    quantized_paths = {
        "ptq_int8": {
            "path": str(QUANTIZED_DIR / "ptq_int8.tflite"),
            "input_range": "0-255",
        },
        "qat_int8": {
            "path": str(QUANTIZED_DIR / "qat_int8.tflite"),
            "input_range": "0-255",
        },
        "mixed_precision": {
            "path": str(QUANTIZED_DIR / "mixed_precision.tflite"),
            "input_range": "0-255",
        },
        "dynamic_range": {
            "path": str(QUANTIZED_DIR / "dynamic_range.tflite"),
            "input_range": "0-255",
        },
    }
    for information in quantized_paths.values():
        if not Path(information["path"]).exists():
            raise FileNotFoundError(
                f"Quantized model not found: {information['path']}"
            )

    _, _, x_test, y_test = load_and_preprocess_data()
    memory_optimizations = implement_memory_optimizations(latency_model)
    memory_management = {
        "gradient_checkpointing": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["gradient_checkpointing"],
                input_size=160,
            ),
            "scope": "training activation-memory proxy",
        },
        "model_sharding": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["model_sharding"],
                input_size=160,
            ),
            "scope": "GPU feature extraction + CPU classification head",
        },
        "activation_compression": {
            "implemented": True,
            "single_inference_time_ms": _latency_only(
                memory_optimizations["activation_compression"],
                input_size=160,
            ),
            "scope": "mixed_float16 classification-head activations",
        },
        "optimal_batch_size": memory_optimizations["optimal_batch_size"],
    }

    baseline = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )
    baseline_accuracy = float(baseline["accuracy"])
    energy_metrics = evaluate_keras_model(energy_model, x_test, y_test)
    energy_metrics = {
        **energy_metrics,
        "full_model_accuracy": energy_metrics["accuracy"],
        **early_exit_predictions(energy_model, x_test, y_test),
        "path": str(ENERGY_MODEL_PATH),
        "model_size_mb": ENERGY_MODEL_PATH.stat().st_size / (1024 ** 2),
    }
    architecture_metrics = {
        "latency_optimized": {
            **evaluate_keras_model(latency_model, x_test, y_test),
            "path": str(LATENCY_MODEL_PATH),
            "model_size_mb": (
                LATENCY_MODEL_PATH.stat().st_size / (1024 ** 2)
            ),
        },
        "memory_optimized": {
            **evaluate_keras_model(memory_model, x_test, y_test),
            "path": str(MEMORY_MODEL_PATH),
            "model_size_mb": (
                MEMORY_MODEL_PATH.stat().st_size / (1024 ** 2)
            ),
            "kernel_sparsity": _weight_sparsity(memory_model),
        },
        "energy_optimized": energy_metrics,
    }
    for metrics in architecture_metrics.values():
        metrics["accuracy_drop_vs_baseline"] = (
            baseline_accuracy - metrics["accuracy"]
        )

    quantized_metrics = {}
    for name, information in quantized_paths.items():
        path = Path(information["path"])
        quantized_metrics[name] = {
            **evaluate_tflite_model(
                path,
                x_test,
                y_test,
                normalize_input=information["input_range"] == "-1 to 1",
            ),
            "path": str(path),
            "input_range": information["input_range"],
            "source_model": "latency_optimized",
        }
        quantized_metrics[name]["accuracy_drop_vs_baseline"] = (
            baseline_accuracy - quantized_metrics[name]["accuracy"]
        )

    comparison = {
        "baseline": baseline,
        "architecture_models": architecture_metrics,
        "quantized_models": quantized_metrics,
        "memory_management": memory_management,
    }
    (OUTPUT_DIR / "performance_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    if "--reuse-existing" in sys.argv:
        evaluate_existing_models()
    else:
        main()
