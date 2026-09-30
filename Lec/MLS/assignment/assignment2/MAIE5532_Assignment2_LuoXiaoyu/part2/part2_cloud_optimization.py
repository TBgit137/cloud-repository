import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42
EPOCHS = 50
TEMPERATURE = 4.0
ALPHA = 0.5
LOGICAL_MEMORY_MB = 4096
OUTPUT = HERE / "part2_results"
BASELINE = ROOT / "part1" / "baseline_model.keras"

# Reuse the Part 1 model and data loader so the comparison stays consistent.
sys.path.insert(0, str(ROOT / "part1"))
from part1_baseline import create_baseline_model, load_and_preprocess_data


def write_json(path, value):
    """Write strict JSON results and reject invalid numeric values."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def configure_devices(simulated=False):
    """Configure the GPU before TensorFlow creates models or tensors."""
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("No GPU was detected. Run this script in the configured WSL TensorFlow environment.")
    tf.config.set_visible_devices(physical[0], "GPU")
    if simulated:
        tf.config.set_logical_device_configuration(physical[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=LOGICAL_MEMORY_MB),
            tf.config.LogicalDeviceConfiguration(memory_limit=LOGICAL_MEMORY_MB),
        ])
    else:
        tf.config.experimental.set_memory_growth(physical[0], True)
    # Limit host threads because replicated input pipelines can otherwise oversubscribe the CPU.
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    return tf.config.experimental.get_device_details(physical[0]).get("device_name", "GPU")


class CloudOptimizer:

    def __init__(self, baseline_model_path):
        with tf.device("/CPU:0"):
            self.baseline_model = keras.models.load_model(baseline_model_path, compile=False)
        # All student models start from the same random initialization.
        keras.utils.set_random_seed(SEED)
        with tf.device("/CPU:0"):
            initial = create_baseline_model()
            self.initial_weights = initial.get_weights()

    def _student(self, policy="float32"):
        """Build a student model with the requested numeric policy."""
        def clone_layer(layer):
            config = layer.get_config()
            config["dtype"] = "float32" if layer is self.baseline_model.layers[-1] else policy
            return layer.__class__.from_config(config)

        model = keras.models.clone_model(self.baseline_model, clone_function=clone_layer)
        model.set_weights(self.initial_weights)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        if policy == "mixed_float16":
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def implement_mixed_precision(self):
        """Enable mixed precision training with automatic loss scaling."""
        mixed_precision.set_global_policy("mixed_float16")
        return self._student("mixed_float16")

    def implement_model_parallelism(self, strategy="mirrored"):
        """Create a mirrored training strategy over the available logical GPUs."""
        if strategy != "mirrored":
            raise ValueError("This implementation uses the mirrored strategy; other strategies need separate cluster setup.")
        devices = [device.name for device in tf.config.list_logical_devices("GPU")]
        training_strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            # ReductionToOneDevice works reliably for logical devices on one physical GPU.
            cross_device_ops=tf.distribute.ReductionToOneDevice(),
        )
        with training_strategy.scope():
            model = self._student()
        return model, training_strategy

    def optimize_batch_processing(self, target_batch_size=256):
        """Return the batch processing configuration for gradient accumulation."""
        micro_batch_size = min(64, target_batch_size)
        if target_batch_size <= 0 or target_batch_size % micro_batch_size:
            raise ValueError("The effective batch size must be positive and divisible by the micro batch size.")
        return {
            "micro_batch_size": micro_batch_size,
            "effective_batch_size": target_batch_size,
            "accumulation_steps": target_batch_size // micro_batch_size,
            "parallel_map": True,
            "prefetch": True,
        }

    def implement_knowledge_distillation(self):
        """Build the teacher, student, and distillation trainer."""
        layers = [keras.Input(shape=(32, 32, 3))]
        for width in (48, 96, 176):
            for _ in range(2):
                layers.extend([
                    keras.layers.Conv2D(width, 3, padding="same"),
                    keras.layers.BatchNormalization(), keras.layers.ReLU(),
                ])
            layers.append(keras.layers.MaxPooling2D(2))
        layers.extend([
            keras.layers.GlobalAveragePooling2D(), keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"), keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation="softmax", dtype="float32"),
        ])
        teacher = keras.Sequential(layers, name="teacher")
        teacher.compile(optimizer=keras.optimizers.Adam(0.001),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        student = self._student()

        def distillation_training_function(trained_teacher, **kwargs):
            """Return a Keras trainer that fits the student with a frozen teacher."""
            trainer = Distiller(student, trained_teacher, **kwargs)
            trainer.compile(optimizer=keras.optimizers.Adam(0.001))
            return trainer

        return teacher, student, distillation_training_function


class AccumulatingModel(keras.Model):
    """Wrap a model so gradients are accumulated across micro batches."""

    def __init__(self, network, accumulation_steps):
        super().__init__()
        self.network = network
        self.accumulation_steps = accumulation_steps
        self.buffers = [tf.Variable(tf.zeros_like(w), trainable=False)
                        for w in network.trainable_variables]
        self.sample_count = tf.Variable(0.0, trainable=False)
        self.micro_count = tf.Variable(0, trainable=False)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    @tf.function
    def flush(self):
        """Apply accumulated gradients normalized by the number of samples."""
        def apply():
            self.optimizer.apply_gradients(
                [(buffer / self.sample_count, weight)
                 for buffer, weight in zip(self.buffers, self.network.trainable_variables)])
            for buffer in self.buffers:
                buffer.assign(tf.zeros_like(buffer))
            self.sample_count.assign(0.0)
            self.micro_count.assign(0)
            return tf.constant(0)
        return tf.cond(self.sample_count > 0, apply, lambda: tf.constant(0))

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            losses = keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss_sum = tf.reduce_sum(losses)
        gradients = tape.gradient(loss_sum, self.network.trainable_variables)
        for buffer, gradient in zip(self.buffers, gradients):
            buffer.assign_add(gradient)
        self.sample_count.assign_add(tf.cast(tf.shape(labels)[0], tf.float32))
        self.micro_count.assign_add(1)
        tf.cond(self.micro_count >= self.accumulation_steps,
                self.flush, lambda: tf.constant(0))
        self.loss_tracker.update_state(losses)
        self.accuracy_tracker.update_state(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        self.loss_tracker.update_state(
            keras.losses.sparse_categorical_crossentropy(labels, predictions))
        self.accuracy_tracker.update_state(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}


class Distiller(keras.Model):
    """Train a student with hard labels and softened teacher predictions."""

    def __init__(self, student, teacher, temperature=4.0, alpha=0.5):
        super().__init__()
        if temperature <= 0 or not 0 <= alpha <= 1:
            raise ValueError("Temperature must be positive and alpha must be in [0, 1].")
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.temperature = temperature
        self.alpha = alpha
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        # Use logits from the final classifier instead of taking logs of saturated softmax outputs.
        self.student_features = keras.Model(student.input, student.layers[-1].input)
        self.teacher_features = keras.Model(teacher.input, teacher.layers[-1].input)

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def logits(self, features_model, classifier, images, training):
        features = features_model(images, training=training)
        return tf.matmul(features, classifier.kernel) + classifier.bias

    def train_step(self, data):
        images, labels = data
        teacher_logits = self.logits(
            self.teacher_features, self.teacher.layers[-1], images, False)
        temperature = self.temperature
        teacher_log_probs = tf.nn.log_softmax(teacher_logits / temperature)
        teacher_probs = tf.exp(teacher_log_probs)
        with tf.GradientTape() as tape:
            student_logits = self.logits(
                self.student_features, self.student.layers[-1], images, True)
            hard_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.int32), logits=student_logits)
            student_log_probs = tf.nn.log_softmax(student_logits / temperature)
            soft_loss = tf.reduce_sum(
                teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
            losses = self.alpha * hard_loss + (1 - self.alpha) * temperature ** 2 * soft_loss
            loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.loss_tracker.update_state(losses)
        self.accuracy_tracker.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        # Validation uses the student's normal cross entropy for comparison with non-distilled models.
        images, labels = data
        predictions = self.student(images, training=False)
        self.loss_tracker.update_state(
            keras.losses.sparse_categorical_crossentropy(labels, predictions))
        self.accuracy_tracker.update_state(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}


def load_data():
    """Use the same stratified validation split as Part 1."""
    x, y, xt, yt = load_and_preprocess_data()
    rng = np.random.default_rng(SEED)
    train, validation = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        count = max(1, int(len(indices) * 0.1))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    train = rng.permutation(train)
    validation = np.asarray(validation)
    return (x[train], y[train]), (x[validation], y[validation]), (xt, yt)


def datasets(data, batch_size, optimized=True):
    """Build datasets with the same augmentation and optional input pipeline optimization."""
    train, validation, test = data
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED + 1),
    ])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.experimental_deterministic = True
    # Run augmentation on CPU and keep tail batches for all datasets.
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(train).shuffle(len(train[0]), seed=SEED)
        ds = ds.batch(batch_size).map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE if optimized else None)
        ds = ds.with_options(options)
        if optimized:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        val = tf.data.Dataset.from_tensor_slices(validation).batch(128).with_options(options)
        test_ds = tf.data.Dataset.from_tensor_slices(test).batch(128).with_options(options)
    return ds, val, test_ds


def memory_info(reset=False):
    """Collect TensorFlow allocator memory statistics for each logical GPU."""
    result = {}
    for device in tf.config.list_logical_devices("GPU"):
        name = device.name.split("/device:")[-1]
        try:
            if reset:
                tf.config.experimental.reset_memory_stats(name)
            result[name] = tf.config.experimental.get_memory_info(name)
        except (ValueError, RuntimeError) as error:
            result[name] = {"unavailable": str(error)}
    return result


class ExperimentCallback(keras.callbacks.Callback):
    """Track training time and save the model with the best validation loss."""

    def __init__(self, export_model, output_dir):
        super().__init__()
        self.export_model = export_model
        self.output_dir = output_dir
        self.best_loss = float("inf")
        self.best_epoch = None
        self.epoch_train_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_test_begin(self, logs=None):
        # Flush the final partial accumulation group before validation.
        if isinstance(self.model, AccumulatingModel):
            self.model.flush()
        self.epoch_train_seconds.append(time.perf_counter() - self.start)

    def on_epoch_end(self, epoch, logs=None):
        if not all(np.isfinite(float(v)) for v in logs.values()):
            raise FloatingPointError("Non-finite training metrics were detected; stopping this experiment.")
        if logs["val_loss"] < self.best_loss:
            self.best_loss = float(logs["val_loss"])
            self.best_epoch = epoch + 1
            # Save an uncompiled network so optimizer and trainer state are excluded.
            with tf.device("/CPU:0"):
                snapshot = keras.models.clone_model(self.export_model)
                snapshot.set_weights(self.export_model.get_weights())
                snapshot.save(self.output_dir / "best_model.keras")


def inference_benchmark(model, images):
    """Measure single-image inference latency after warmup."""
    sample = tf.convert_to_tensor(images[:1])

    @tf.function
    def infer(x):
        return model(x, training=False)

    for _ in range(20):
        infer(sample).numpy()
    start = time.perf_counter()
    for _ in range(100):
        infer(sample).numpy()
    return (time.perf_counter() - start) * 1000 / 100


# The Float32 model is reused as the control for multiple comparisons.
GROUPS = {
    "mixed": ["fp32", "mixed"],
    "distributed": ["distributed_one", "distributed_two"],
    "batch": ["pipeline_serial", "fp32", "batch256", "accumulation"],
    "distillation": ["fp32", "teacher", "distilled"],
}
VARIANTS = list(dict.fromkeys(name for group in GROUPS.values() for name in group))


def run_experiment(name):
    """Run one configuration and record accuracy, latency, memory, and throughput."""
    simulated = name.startswith("distributed_")
    gpu_name = configure_devices(simulated)
    mixed_precision.set_global_policy("float32")
    keras.utils.set_random_seed(SEED)
    optimizer = CloudOptimizer(BASELINE)
    batch_size, replicas, extra = 128, 1, {}
    if name == "mixed":
        model = trainer = optimizer.implement_mixed_precision()
    elif name == "distributed_two":
        model, strategy = optimizer.implement_model_parallelism()
        trainer, replicas = model, strategy.num_replicas_in_sync
    elif name == "distributed_one":
        # The one-replica control uses the same logical-device memory limit.
        with tf.distribute.OneDeviceStrategy("/GPU:0").scope():
            model = trainer = optimizer._student()
    elif name in ("teacher", "distilled"):
        teacher, student, distillation_training = optimizer.implement_knowledge_distillation()
        extra["teacher_parameter_ratio"] = teacher.count_params() / student.count_params()
        if name == "teacher":
            model = trainer = teacher
        else:
            teacher_path = OUTPUT / "cloud_optimized_models" / "teacher" / "best_model.keras"
            teacher.set_weights(keras.models.load_model(teacher_path, compile=False).get_weights())
            model = student
            trainer = distillation_training(teacher, temperature=TEMPERATURE, alpha=ALPHA)
            extra.update(temperature=TEMPERATURE, alpha=ALPHA)
    else:
        model = trainer = optimizer._student()
        if name == "batch256":
            batch_size = 256
        elif name == "accumulation":
            config = optimizer.optimize_batch_processing(256)
            batch_size = config["micro_batch_size"]
            trainer = AccumulatingModel(model, config["accumulation_steps"])
            trainer.compile(optimizer=keras.optimizers.Adam(0.001))

    keras.utils.set_random_seed(SEED)
    data = load_data()
    train_ds, val_ds, test_ds = datasets(data, batch_size, name != "pipeline_serial")
    folder = OUTPUT / "cloud_optimized_models" / name
    folder.mkdir(parents=True, exist_ok=True)
    callback = ExperimentCallback(model, folder)
    # Use the same epoch count and learning-rate schedule for speed comparisons.
    def learning_rate(epoch):
        return 0.001 if epoch < EPOCHS * 0.5 else (0.0005 if epoch < EPOCHS * 0.8 else 0.00025)

    memory_info(reset=True)
    start = time.perf_counter()
    history = trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
                          callbacks=[callback, keras.callbacks.LearningRateScheduler(learning_rate)])
    training_seconds = time.perf_counter() - start
    training_memory = memory_info()
    updates = int(trainer.optimizer.iterations.numpy())
    model = keras.models.load_model(folder / "best_model.keras", compile=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    test = model.evaluate(test_ds, verbose=0, return_dict=True)
    # The first epoch includes compilation overhead, so steady-state throughput excludes it.
    durations = callback.epoch_train_seconds[1:]
    metrics = {
        "epochs": EPOCHS, "best_epoch": callback.best_epoch,
        "train_examples": len(data[0][0]), "test_examples": len(data[2][0]),
        "test_accuracy": float(test["accuracy"]), "test_loss": float(test["loss"]),
        "parameters": model.count_params(),
        "model_size_bytes": (folder / "best_model.keras").stat().st_size,
        "training_seconds": training_seconds,
        "train_images_per_second": len(data[0][0]) / float(np.mean(durations)) if durations else None,
        "training_memory": training_memory,
        "inference_mean_ms": inference_benchmark(model, data[2][0]),
        "micro_batch_size": batch_size,
        "effective_batch_size": 256 if name == "accumulation" else batch_size,
        "optimizer_updates": updates, "simulated": simulated, "replicas": replicas,
        "gpu_name": gpu_name, "tensorflow_version": tf.__version__, "seed": SEED,
        **extra,
    }
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "history.json", {k: [float(v) for v in values]
                                         for k, values in history.history.items()})
    return metrics


def save_report(results):
    """Save the summary JSON report and the comparison figure."""
    comparisons = {}
    for reference, candidate in [("fp32", "mixed"), ("pipeline_serial", "fp32"),
                                  ("batch256", "accumulation"), ("fp32", "distilled"),
                                  ("distributed_one", "distributed_two")]:
        if reference in results and candidate in results:
            a, b = results[reference], results[candidate]
            speed = b["train_images_per_second"] / a["train_images_per_second"] if a["train_images_per_second"] and b["train_images_per_second"] else None
            comparisons[f"{candidate}_vs_{reference}"] = {
                "throughput_ratio": speed,
                "accuracy_change_percentage_points": 100 * (b["test_accuracy"] - a["test_accuracy"]),
            }
            if candidate == "distributed_two":
                comparisons[f"{candidate}_vs_{reference}"]["simulated_efficiency"] = speed / 2 if speed else None
    write_json(OUTPUT / "cloud_optimization_report.json", {
        "results": results, "comparisons": comparisons,
        "notes": [
            "Training throughput excludes the first epoch, validation, and saving; total training time includes those costs.",
            "Memory values are TensorFlow allocator statistics, not total process GPU memory; logical device values should not be added together.",
            "Two logical replicas on one physical GPU are used only for simulation and do not represent real multi-GPU scaling.",
            "Saved model files exclude optimizer state; inference latency includes output transfer but excludes loading and preprocessing.",
        ],
    })
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, [results[n]["test_accuracy"] * 100 for n in names])
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].bar(names, [results[n]["train_images_per_second"] or 0 for n in names])
    axes[1].set_ylabel("Training images/s")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=65)
    fig.tight_layout()
    fig.savefig(OUTPUT / "performance_comparison.png", dpi=160)
    plt.close(fig)


def benchmark_cloud_optimizations(experiment="all"):
    if not BASELINE.is_file():
        raise FileNotFoundError(f"Part 1 model was not found: {BASELINE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = VARIANTS if experiment == "all" else GROUPS[experiment]
    results = {}
    for name in names:
        print(f"Running: {name}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name], check=True)
        path = OUTPUT / "cloud_optimized_models" / name / "metrics.json"
        results[name] = json.loads(path.read_text(encoding="utf-8"))
    save_report(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker" and sys.argv[2] in VARIANTS:
        run_experiment(sys.argv[2])
    else:
        group = sys.argv[1] if len(sys.argv) == 2 else "all"
        if len(sys.argv) > 2 or group not in ["all", *GROUPS]:
            raise SystemExit("Usage: python part2_cloud_optimization.py [all|mixed|distributed|batch|distillation]")
        results = benchmark_cloud_optimizations(group)
        for name, metrics in results.items():
            print(f"{name}: accuracy={metrics['test_accuracy']:.2%}, training time={metrics['training_seconds']:.2f}s")
        print(f"Results directory: {OUTPUT}")
