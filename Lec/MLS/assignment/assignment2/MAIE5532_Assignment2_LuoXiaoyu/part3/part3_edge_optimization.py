import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras


SEED = 42
BATCH_SIZE = 128
PRUNE_EPOCHS = 5
NAS_SEARCH_EPOCHS = 4
ARCH_FINAL_EPOCHS = 20
RESULT_EVAL_SAMPLES = 10000
LATENCY_RUNS = 100

HERE = Path(__file__).resolve().parent
PART1_DIR = HERE.parent / "part1"
BASELINE_MODEL_PATH = PART1_DIR / "baseline_model.keras"
RESULTS_DIR = HERE / "part3_results"
MODEL_DIR = RESULTS_DIR / "edge_optimized_models"
TFLITE_DIR = RESULTS_DIR / "tflite_models"


def load_and_preprocess_data():
    """Load CIFAR-10 and normalize pixel values to the [0, 1] range."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)


def make_train_val_data(x_train, y_train, batch_size=BATCH_SIZE):
    """Reserve 10% of each class for validation and build batched datasets."""
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for label in np.unique(y_train):
        indices = rng.permutation(np.flatnonzero(y_train == label))
        val_count = max(1, int(len(indices) * 0.1))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_indices = rng.permutation(train_indices)
    val_indices = np.asarray(val_indices)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[train_indices], y_train[train_indices])
    )
    train_ds = train_ds.shuffle(len(train_indices), seed=SEED).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[val_indices], y_train[val_indices])
    ).batch(batch_size)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def compile_model(model, learning_rate=1e-3):
    """Compile a model with Adam, sparse cross-entropy, and accuracy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_weight_zeros(model):
    """Count zeros across all stored weights, including biases and BatchNorm state."""
    total = 0
    zeros = 0
    for weight in model.get_weights():
        total += weight.size
        zeros += int(np.count_nonzero(weight == 0))
    return {
        "zero_weights": zeros,
        "total_weights": total,
        "actual_sparsity": zeros / total if total else 0.0,
    }


def model_weight_bytes(model):
    """Calculate weight storage in bytes, excluding activations and runtime overhead."""
    return int(sum(weight.size * weight.dtype.itemsize for weight in model.get_weights()))


def save_keras_model_size(model, name):
    """Save a Keras model and return its path and file size in bytes."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.keras"
    model.save(path)
    return path, os.path.getsize(path)


def gzip_size(path):
    """Return the gzip-compressed file size to compare model compressibility."""
    import gzip
    with open(path, "rb") as source:
        data = source.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as target:
        target.write(gzip.compress(data))
        gz_path = target.name
    size = os.path.getsize(gz_path)
    os.remove(gz_path)
    return size


def create_edge_candidate(depth, width_multiplier, kernel_size, name):
    """Build a lightweight separable CNN with configurable depth, width, and kernel size."""
    filters = [max(8, int(v * width_multiplier)) for v in (32, 64, 128)[:depth]]
    model = keras.Sequential(name=name)
    model.add(keras.Input(shape=(32, 32, 3)))
    for block_index, filters_count in enumerate(filters):
        # Use depthwise and pointwise convolutions to reduce the parameter count.
        model.add(keras.layers.SeparableConv2D(
            filters_count, kernel_size, padding="same", use_bias=False,
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU(max_value=6.0))
        model.add(keras.layers.SeparableConv2D(
            filters_count, kernel_size, padding="same", use_bias=False,
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU(max_value=6.0))
        if block_index < len(filters) - 1:
            model.add(keras.layers.MaxPooling2D((2, 2)))

    # Reduce each feature map to its maximum value before classification.
    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(max(32, int(128 * width_multiplier)), activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return compile_model(model)


def representative_dataset(x_train, count=100):
    """Yield training images for INT8 calibration without using test samples."""
    for image in x_train[:count]:
        yield [np.expand_dims(image.astype(np.float32), axis=0)]


def convert_to_tflite(model, name, x_train, mode="float32"):
    """Convert and save a Keras model using the specified TFLite precision mode."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(x_train)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif mode != "float32":
        raise ValueError(f"Unknown TFLite mode: {mode}")

    model_bytes = converter.convert()
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    path = TFLITE_DIR / f"{name}.tflite"
    path.write_bytes(model_bytes)
    return path


def run_tflite(interpreter, image):
    """Run single-image inference with integer input quantization and output dequantization."""
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_data = np.expand_dims(image, axis=0)

    if input_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = input_detail["quantization"]
        input_data = input_data / scale + zero_point
        input_data = np.clip(
            np.round(input_data),
            np.iinfo(input_detail["dtype"]).min,
            np.iinfo(input_detail["dtype"]).max,
        ).astype(input_detail["dtype"])
    else:
        input_data = input_data.astype(input_detail["dtype"])

    interpreter.set_tensor(input_detail["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    if output_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_detail["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    return output


def evaluate_tflite(path, x_test, y_test, samples=RESULT_EVAL_SAMPLES):
    """Measure TFLite accuracy and batch-size-one latency on a single CPU thread."""
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()

    sample_count = min(samples, len(x_test))
    correct = 0
    for image, label in zip(x_test[:sample_count], y_test[:sample_count]):
        prediction = run_tflite(interpreter, image)
        correct += int(np.argmax(prediction) == int(label))

    warmup = min(20, sample_count)
    for image in x_test[:warmup]:
        run_tflite(interpreter, image)

    durations = []
    timing_count = min(LATENCY_RUNS, sample_count)
    for image in x_test[:timing_count]:
        start = time.perf_counter()
        run_tflite(interpreter, image)
        durations.append((time.perf_counter() - start) * 1000)

    return {
        "accuracy": correct / sample_count,
        "accuracy_samples": sample_count,
        "latency_mean_ms": float(np.mean(durations)),
        "latency_median_ms": float(np.median(durations)),
        "latency_p95_ms": float(np.percentile(durations, 95)),
        "latency_runs": timing_count,
    }


class EdgeOptimizer:
    """Build, convert, and benchmark edge model variants from a trained baseline."""

    def __init__(self, baseline_model_path):
        self.baseline_model_path = Path(baseline_model_path)
        self.baseline_model = keras.models.load_model(self.baseline_model_path)
        self.x_train, self.y_train, self.x_test, self.y_test = load_and_preprocess_data()
        self.train_ds, self.val_ds = make_train_val_data(self.x_train, self.y_train)

    def implement_pruning(self, target_sparsity=0.75):
        """Fine-tune with magnitude pruning and return the model without pruning wrappers."""
        model = keras.models.clone_model(self.baseline_model)
        model.set_weights(self.baseline_model.get_weights())
        steps_per_epoch = int(np.ceil(45000 / BATCH_SIZE))
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=steps_per_epoch * PRUNE_EPOCHS,
            )
        }
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        compile_model(pruned_model, learning_rate=1e-4)
        pruned_model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=PRUNE_EPOCHS,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
            verbose=2,
        )
        stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        compile_model(stripped_model, learning_rate=1e-4)
        return stripped_model

    def implement_quantization(self):
        """Export dynamic range, INT8, and Float16 variants of the baseline."""
        quantized_models = {}
        for mode in ("dynamic_range", "int8", "float16"):
            path = convert_to_tflite(self.baseline_model, f"baseline_{mode}", self.x_train, mode)
            quantized_models[mode] = {
                "type": "tflite",
                "path": str(path),
                "mode": mode,
            }
        return quantized_models

    def implement_architecture_optimization(self):
        """Train the fixed lightweight architecture from scratch."""
        model = create_edge_candidate(
            depth=3, width_multiplier=0.5, kernel_size=3,
            name="edge_architecture_optimized",
        )
        model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=ARCH_FINAL_EPOCHS,
            verbose=2,
        )
        return model

    def implement_neural_architecture_search(self):
        """Search eight configurations and return the retrained selection and search results."""
        search_space = []
        for depth in (2, 3):
            for width_multiplier in (0.5, 0.75):
                for kernel_size in (3, 5):
                    search_space.append((depth, width_multiplier, kernel_size))

        search_results = []
        best_score = -np.inf
        best_config = None
        for depth, width_multiplier, kernel_size in search_space:
            name = f"nas_d{depth}_w{str(width_multiplier).replace('.', '')}_k{kernel_size}"
            model = create_edge_candidate(depth, width_multiplier, kernel_size, name)
            history = model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=NAS_SEARCH_EPOCHS,
                verbose=2,
            )
            val_accuracy = float(max(history.history["val_accuracy"]))
            params = int(model.count_params())
            # Rank candidates by validation accuracy with a parameter-count penalty.
            score = val_accuracy - params / 1_000_000
            result = {
                "name": name,
                "depth": depth,
                "width_multiplier": width_multiplier,
                "kernel_size": kernel_size,
                "validation_accuracy": val_accuracy,
                "parameters": params,
                "score": score,
            }
            search_results.append(result)
            if score > best_score:
                best_score = score
                best_config = (depth, width_multiplier, kernel_size, name)

        best_depth, best_width, best_kernel, best_name = best_config
        best_architecture = create_edge_candidate(
            best_depth, best_width, best_kernel, f"{best_name}_final",
        )
        best_architecture.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=ARCH_FINAL_EPOCHS,
            verbose=2,
        )
        return best_architecture, search_results

    def create_tflite_models(self, models_dict):
        """Export Float32 TFLite models and record their sizes, accuracies, and latencies."""
        results = {}
        for name, model in models_dict.items():
            path = convert_to_tflite(model, name, self.x_train, "float32")
            metrics = evaluate_tflite(path, self.x_test, self.y_test)
            results[name] = {
                "path": str(path),
                "size_bytes": os.path.getsize(path),
                "size_mib": os.path.getsize(path) / (1024 ** 2),
                **metrics,
            }
        return results

    def benchmark_keras_model(self, model, name):
        """Record Keras accuracy, parameter count, weight storage, and saved file sizes."""
        eval_samples = min(RESULT_EVAL_SAMPLES, len(self.x_test))
        loss, accuracy = model.evaluate(
            self.x_test[:eval_samples],
            self.y_test[:eval_samples],
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        path, size_bytes = save_keras_model_size(model, name)
        return {
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "test_samples": eval_samples,
            "parameters": int(model.count_params()),
            "weight_memory_bytes": model_weight_bytes(model),
            "keras_file_bytes": size_bytes,
            "keras_gzip_bytes": gzip_size(path),
            **count_weight_zeros(model),
        }

    def estimate_edge_viability(self, tflite_results):
        """Estimate platform feasibility using model size and local TFLite latency."""
        platforms = {
            "microcontroller_1mb_flash": {"flash_bytes": 1 * 1024 ** 2, "ram_bytes": 256 * 1024, "power_mw": 50},
            "small_linux_edge_64mb": {"flash_bytes": 64 * 1024 ** 2, "ram_bytes": 64 * 1024 ** 2, "power_mw": 1000},
            "raspberry_pi_class": {"flash_bytes": 512 * 1024 ** 2, "ram_bytes": 512 * 1024 ** 2, "power_mw": 3000},
        }
        estimates = {}
        for model_name, metrics in tflite_results.items():
            estimates[model_name] = {}
            for platform_name, spec in platforms.items():
                size_ok = metrics["size_bytes"] <= spec["flash_bytes"]
                # Use twice the file size as a RAM heuristic, not a measured activation-memory requirement.
                estimated_ram = metrics["size_bytes"] * 2
                ram_ok = estimated_ram <= spec["ram_bytes"]
                energy_mj = metrics["latency_mean_ms"] * spec["power_mw"] / 1000
                estimates[model_name][platform_name] = {
                    "fits_flash": bool(size_ok),
                    "estimated_ram_ok": bool(ram_ok),
                    "estimated_energy_mj_per_inference": float(energy_mj),
                    "note": "Estimated from local TFLite latency, not measured on this hardware.",
                }
        return estimates


def benchmark_edge_optimizations():
    """Benchmark edge optimization strategies and save their results as JSON."""
    keras.utils.set_random_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optimizer = EdgeOptimizer(BASELINE_MODEL_PATH)
    results = {
        "settings": {
            "seed": SEED,
            "prune_epochs": PRUNE_EPOCHS,
            "nas_search_epochs": NAS_SEARCH_EPOCHS,
            "architecture_final_epochs": ARCH_FINAL_EPOCHS,
            "tflite_accuracy_samples": RESULT_EVAL_SAMPLES,
            "latency_runs": LATENCY_RUNS,
            "note": "No physical edge board was available; viability is estimated from model size and local TFLite CPU latency.",
        }
    }

    baseline_tflite = convert_to_tflite(
        optimizer.baseline_model, "baseline_float32", optimizer.x_train, "float32",
    )
    results["baseline"] = {
        "keras": optimizer.benchmark_keras_model(optimizer.baseline_model, "baseline"),
        "tflite": {
            "path": str(baseline_tflite),
            "size_bytes": os.path.getsize(baseline_tflite),
            "size_mib": os.path.getsize(baseline_tflite) / (1024 ** 2),
            **evaluate_tflite(baseline_tflite, optimizer.x_test, optimizer.y_test),
        },
    }

    pruning_results = {}
    pruned_models = {}
    for sparsity in (0.50, 0.75):
        name = f"pruned_{int(sparsity * 100)}"
        model = optimizer.implement_pruning(target_sparsity=sparsity)
        pruned_models[name] = model
        pruning_results[name] = optimizer.benchmark_keras_model(model, name)
    results["pruning"] = pruning_results

    quantization_results = {}
    for mode, metadata in optimizer.implement_quantization().items():
        path = Path(metadata["path"])
        quantization_results[mode] = {
            "path": str(path),
            "size_bytes": os.path.getsize(path),
            "size_mib": os.path.getsize(path) / (1024 ** 2),
            **evaluate_tflite(path, optimizer.x_test, optimizer.y_test),
        }
    results["quantization"] = quantization_results

    architecture_model = optimizer.implement_architecture_optimization()
    best_nas_model, search_results = optimizer.implement_neural_architecture_search()
    architecture_models = {
        "architecture_optimized": architecture_model,
        "nas_best": best_nas_model,
    }
    results["architecture"] = {
        name: optimizer.benchmark_keras_model(model, name)
        for name, model in architecture_models.items()
    }
    results["nas_search"] = search_results

    all_keras_models = {**pruned_models, **architecture_models}
    optimized_tflite = optimizer.create_tflite_models(all_keras_models)
    results["tflite_conversions"] = optimized_tflite

    edge_model_results = {
        "baseline_float32": results["baseline"]["tflite"],
        **quantization_results,
        **optimized_tflite,
    }
    results["edge_viability"] = optimizer.estimate_edge_viability(edge_model_results)

    output_path = RESULTS_DIR / "edge_optimization_report.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    results = benchmark_edge_optimizations()
    print("Edge Optimization Results:")
    print(json.dumps(results, indent=2))
