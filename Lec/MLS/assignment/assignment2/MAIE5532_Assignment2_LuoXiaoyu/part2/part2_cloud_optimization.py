"""第二部分：混合精度、分布式模拟、批处理优化和知识蒸馏。

在提交目录运行：python part2/part2_cloud_optimization.py
可追加 mixed、distributed、batch 或 distillation，仅运行对应实验组。
使用 WSL / Python 3.11 / TensorFlow 2.15.1。
"""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

# 本部分没有使用剪枝或量化 API，因此不引入题目示例中未使用的 tfmot。
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 42
EPOCHS = 50
TEMPERATURE = 4.0
ALPHA = 0.5
LOGICAL_MEMORY_MB = 4096
OUTPUT = HERE / "part2_results"
BASELINE = ROOT / "part1" / "baseline_model.keras"

# 复用第一部分的模型定义和数据加载，不修改第一部分文件。
sys.path.insert(0, str(ROOT / "part1"))
from part1_baseline import create_baseline_model, load_and_preprocess_data


def write_json(path, value):
    """使用标准 JSON 保存结果；拒绝把 NaN 当成有效实验指标。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def configure_devices(simulated=False):
    """必须在创建模型或张量前配置设备；模拟使用一张物理 GPU。"""
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("未识别 GPU，请在已配置好的 WSL TensorFlow 环境运行。")
    tf.config.set_visible_devices(physical[0], "GPU")
    if simulated:
        tf.config.set_logical_device_configuration(physical[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=LOGICAL_MEMORY_MB),
            tf.config.LogicalDeviceConfiguration(memory_limit=LOGICAL_MEMORY_MB),
        ])
    else:
        tf.config.experimental.set_memory_growth(physical[0], True)
    # 限制线程池，避免多副本数据管线创建过多宿主线程。
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    return tf.config.experimental.get_device_details(physical[0]).get("device_name", "GPU")


class CloudOptimizer:
    """保留作业的类与四个优化接口，基线权重只用于读取和参照。"""

    def __init__(self, baseline_model_path):
        with tf.device("/CPU:0"):
            self.baseline_model = keras.models.load_model(baseline_model_path, compile=False)
        # 所有学生对照都从同一份随机初始化开始，而非从已训练模型微调。
        keras.utils.set_random_seed(SEED)
        with tf.device("/CPU:0"):
            initial = create_baseline_model()
            self.initial_weights = initial.get_weights()

    def _student(self, policy="float32"):
        """按基线配置重建网络；逐层指定策略，输出概率始终为 Float32。"""
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
        """启用混合精度；权重保持 Float32，训练由 Keras 自动处理损失缩放。"""
        mixed_precision.set_global_policy("mixed_float16")
        return self._student("mixed_float16")

    def implement_model_parallelism(self, strategy="mirrored"):
        """保留题目方法名，实际实现单机同步数据并行，而非切分模型。

        多工作节点和参数服务器需要额外集群，本次采用确认过的 mirrored。
        逻辑 GPU 模拟只能验证同步流程，不能证明真实多 GPU 的扩展能力。
        """
        if strategy != "mirrored":
            raise ValueError("本次实现采用 mirrored；其他策略需要独立集群配置。")
        devices = [device.name for device in tf.config.list_logical_devices("GPU")]
        training_strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            # 使用归并到单设备的归约方式，兼容同一物理卡上的逻辑副本。
            cross_device_ops=tf.distribute.ReductionToOneDevice(),
        )
        with training_strategy.scope():
            model = self._student()
        return model, training_strategy

    def optimize_batch_processing(self, target_batch_size=256):
        """返回有效批次配置，包含梯度累积、并行映射和预取设置。"""
        micro_batch_size = min(64, target_batch_size)
        if target_batch_size <= 0 or target_batch_size % micro_batch_size:
            raise ValueError("有效批大小必须为正，且为微批大小的整数倍。")
        return {
            "micro_batch_size": micro_batch_size,
            "effective_batch_size": target_batch_size,
            "accumulation_steps": target_batch_size // micro_batch_size,
            "parallel_map": True,
            "prefetch": True,
        }

    def implement_knowledge_distillation(self):
        """教师使用约两倍基线参数；学生架构沿用基线，返回蒸馏训练入口。"""
        layers = [keras.Input(shape=(32, 32, 3))]
        # 不是把通道直接翻倍：卷积参数随输入和输出通道的乘积增长。
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
            """接收训练好的教师，返回可调用 fit 的蒸馏训练器。"""
            trainer = Distiller(student, trained_teacher, **kwargs)
            trainer.compile(optimizer=keras.optimizers.Adam(0.001))
            return trainer

        return teacher, student, distillation_training_function


class AccumulatingModel(keras.Model):
    """按样本数累加梯度，最后不足一个有效批次的数据也会更新参数。"""

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
        """按累计样本数归一化，不把最后一个不足批次当成完整批次。"""
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
    """监督交叉熵与温度软化后的 KL 散度联合训练，教师始终保持推理模式。"""

    def __init__(self, student, teacher, temperature=4.0, alpha=0.5):
        super().__init__()
        if temperature <= 0 or not 0 <= alpha <= 1:
            raise ValueError("温度必须大于零，监督损失权重必须在 [0, 1] 内。")
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.temperature = temperature
        self.alpha = alpha
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        # 直接提取最后一层的 logits，避免对已饱和的 softmax 概率取对数。
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
        # 验证损失只用学生的普通交叉熵，便于和非蒸馏学生比较。
        images, labels = data
        predictions = self.student(images, training=False)
        self.loss_tracker.update_state(
            keras.losses.sparse_categorical_crossentropy(labels, predictions))
        self.accuracy_tracker.update_state(labels, predictions)
        return {metric.name: metric.result() for metric in self.metrics}


def load_data():
    """使用与 Part 1 完全相同的分层验证划分。"""
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
    """两种管线都保留相同增强，仅改变并行映射与预取。"""
    train, validation, test = data
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED + 1),
    ])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.experimental_deterministic = True
    # 增强在 CPU 上执行，减少设备内存占用；三种数据集都不丢弃尾批。
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
    """记录 TensorFlow 分配器统计；不可用时明确记录原因，不填写零值。"""
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
    """分别记录训练批次耗时和总耗时，保存验证损失最低的可部署模型。"""

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
        # fit 验证前处理最后一个累积组，计入训练耗时。
        if isinstance(self.model, AccumulatingModel):
            self.model.flush()
        self.epoch_train_seconds.append(time.perf_counter() - self.start)

    def on_epoch_end(self, epoch, logs=None):
        if not all(np.isfinite(float(v)) for v in logs.values()):
            raise FloatingPointError("出现非有限训练指标，停止实验以避免保存无效结果。")
        if logs["val_loss"] < self.best_loss:
            self.best_loss = float(logs["val_loss"])
            self.best_epoch = epoch + 1
            # 导出未编译的普通网络，统一排除优化器和训练器状态。
            with tf.device("/CPU:0"):
                snapshot = keras.models.clone_model(self.export_model)
                snapshot.set_weights(self.export_model.get_weights())
                snapshot.save(self.output_dir / "best_model.keras")


def inference_benchmark(model, images):
    """预热后测量单张推理耗时；取回输出以等待 GPU 计算完成。"""
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


# 普通 Float32 模型同时作为混合精度、批处理和蒸馏的对照。
GROUPS = {
    "mixed": ["fp32", "mixed"],
    "distributed": ["distributed_one", "distributed_two"],
    "batch": ["pipeline_serial", "fp32", "batch256", "accumulation"],
    "distillation": ["fp32", "teacher", "distilled"],
}
VARIANTS = list(dict.fromkeys(name for group in GROUPS.values() for name in group))


def run_experiment(name):
    """运行一项配置，记录作业要求的训练、内存、准确率和吞吐指标。"""
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
        # 单副本对照也使用相同逻辑设备划分，保持显存配额一致。
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
    # 各配置使用同样的固定轮数和学习率时间表，防止早停影响速度比较。
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
    # 第一轮含编译开销；只训练一轮时不报告稳态吞吐量。
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
    """保存一份汇总 JSON 和对比图；书面分析依据正式实验结果编写。"""
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
        "notes": ["训练吞吐排除第一轮、验证和保存；总训练耗时包含这些开销。",
                  "内存为 TensorFlow 分配器字节数，不是进程总显存；逻辑设备统计不可相加。",
                  "单张 GPU 的双逻辑副本只用于模拟，不代表真实双卡扩展效率。",
                  "模型文件不含优化器；推理耗时包含输出回传，不含加载和预处理。"],
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
    """独立进程隔离 GPU 配置及显存统计，按题目接口返回实验结果。"""
    if not BASELINE.is_file():
        raise FileNotFoundError(f"找不到 Part 1 模型：{BASELINE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = VARIANTS if experiment == "all" else GROUPS[experiment]
    results = {}
    for name in names:
        print(f"正在运行：{name}", flush=True)
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
            raise SystemExit("用法：python part2_cloud_optimization.py [all|mixed|distributed|batch|distillation]")
        results = benchmark_cloud_optimizations(group)
        for name, metrics in results.items():
            print(f"{name}: 准确率={metrics['test_accuracy']:.2%}，训练耗时={metrics['training_seconds']:.2f}s")
        print(f"结果目录：{OUTPUT}")
