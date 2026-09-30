import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "part4_results"
MODEL_DIR = OUTPUT / "optimized_models"
SEED = 42
LATENCY_RUNS = 100
MB = 1024 ** 2  # Convert byte counts to MiB for all size and memory fields.


@dataclass
class DeploymentTarget:
    """Define resource limits and the optimizer category for a deployment target."""
    name: str
    max_model_size_mb: float
    max_latency_ms: float
    max_memory_mb: float
    power_budget_mw: float
    compute_capability: str


@dataclass
class OptimizationResult:
    """Store optimization results; use None when runtime memory has not been measured."""
    model_path: str
    accuracy: float
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: Optional[float]
    optimization_strategy: str
    details: Dict[str, Any] = field(default_factory=dict)


def load_json(path):
    """Read a UTF-8 benchmark report."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_data():
    """Rebuild the validation split; calibrate on training data and reserve test data for final evaluation."""
    (images, labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    images = images.astype(np.float32) / 255.0
    labels = labels.reshape(-1)
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        split = int(len(indices) * 0.1)
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])
    # Shuffle the class-grouped indices before selecting calibration samples.
    rng.shuffle(train_indices)
    return (images[np.asarray(train_indices[:100])],
            images[np.asarray(val_indices)], labels[np.asarray(val_indices)],
            test_images.astype(np.float32) / 255.0, test_labels.reshape(-1))


def convert_tflite(model, path, mode, calibration):
    """Apply post-training quantization with integer inputs, operations, and outputs in INT8 mode."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "int8":
        def representative_data():
            for image in calibration:
                yield [image[None, ...]]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif mode == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif mode != "dynamic_range":
        raise ValueError(f"Unsupported quantization mode: {mode}")
    path.write_bytes(converter.convert())


def evaluate_tflite(path, images, labels):
    """Evaluate on one CPU thread; timing includes I/O handling but excludes loading and normalization."""
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    def predict(image):
        value = image[None, ...]
        if np.issubdtype(input_info["dtype"], np.integer):
            scale, zero = input_info["quantization"]
            limits = np.iinfo(input_info["dtype"])
            value = np.clip(np.round(value / scale + zero), limits.min, limits.max)
        interpreter.set_tensor(input_info["index"], value.astype(input_info["dtype"]))
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])
        if np.issubdtype(output_info["dtype"], np.integer):
            scale, zero = output_info["quantization"]
            output = (output.astype(np.float32) - zero) * scale
        return int(np.argmax(output))

    correct = sum(predict(image) == int(label) for image, label in zip(images, labels))
    for image in images[:20]:
        predict(image)
    durations = []
    for image in images[:LATENCY_RUNS]:
        start = time.perf_counter()
        predict(image)
        durations.append((time.perf_counter() - start) * 1000)
    return {"accuracy": correct / len(labels), "samples": len(labels),
            "latency_mean_ms": float(np.mean(durations)),
            "latency_runs": len(durations)}


class ModelOptimizer(ABC):
    """Provide a shared optimization interface for all deployment targets."""
    @abstractmethod
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        pass


class CloudOptimizer(ModelOptimizer):
    def optimize(self, model, target):
        """Select the highest-throughput cloud configuration within measured resource limits."""
        report = load_json(ROOT / "part2/part2_results/cloud_optimization_report.json")
        candidates = []
        for name, metrics in report["results"].items():
            # Exclude simulated configurations from hardware performance selection.
            if metrics["simulated"]:
                continue
            source = ROOT / "part2/part2_results/cloud_optimized_models" / name / "best_model.keras"
            size = source.stat().st_size / MB
            peaks = [entry["peak"] / MB for entry in metrics["training_memory"].values()]
            if (size <= target.max_model_size_mb
                    and metrics["inference_mean_ms"] <= target.max_latency_ms
                    and (not peaks or max(peaks) <= target.max_memory_mb)):
                candidates.append((name, metrics, source))
        if not candidates:
            raise ValueError("No cloud candidate meets the measured resource limits.")
        name, metrics, source = max(candidates, key=lambda item: item[1]["train_images_per_second"])
        destination = MODEL_DIR / "cloud_server.keras"
        shutil.copy2(source, destination)
        return OptimizationResult(
            str(destination), metrics["test_accuracy"], destination.stat().st_size / MB,
            metrics["inference_mean_ms"], None, f"Reused cloud configuration: {name}",
            {"selected_candidate": name,
             "selection_basis": "Highest training throughput within measured limits; not highest inference throughput",
             "train_images_per_second": metrics["train_images_per_second"],
             "training_memory": metrics["training_memory"],
             "latency_scope": "Previously measured local GPU single-image inference, including host output transfer",
             "candidate_throughput": {n: m["train_images_per_second"] for n, m, _ in candidates}},
        )


class EdgeOptimizer(ModelOptimizer):
    def __init__(self, data):
        self.calibration, self.x_val, self.y_val, self.x_test, self.y_test = data

    def candidates(self, model):
        """Build candidates for the three edge quantization modes."""
        return [(mode, model, mode) for mode in ("dynamic_range", "int8", "float16")]

    def optimize(self, model, target):
        """Select using validation data, then evaluate the final model on the independent test set."""
        results = []
        for name, candidate, mode in self.candidates(model):
            print(f"  Converting and evaluating candidate: {target.name}/{name}", flush=True)
            path = MODEL_DIR / f"{target.name}_{name}.tflite"
            convert_tflite(candidate, path, mode, self.calibration)
            metrics = evaluate_tflite(path, self.x_val, self.y_val)
            results.append({"name": name, "path": str(path),
                            "size_mb": path.stat().st_size / MB, **metrics})
        feasible = [r for r in results if r["size_mb"] <= target.max_model_size_mb
                    and r["latency_mean_ms"] <= target.max_latency_ms]
        if not feasible:
            raise ValueError(f"{target.name} has no candidate meeting file-size and local reference latency limits.")
        if target.compute_capability == "tiny":
            # Prefer the smallest TinyML file; break size ties using validation accuracy.
            selected = min(feasible, key=lambda r: (r["size_mb"], -r["accuracy"]))
        else:
            selected = max(feasible, key=lambda r: (r["accuracy"], -r["size_mb"]))
        destination = MODEL_DIR / f"{target.name}.tflite"
        shutil.copy2(selected["path"], destination)
        test = evaluate_tflite(destination, self.x_test, self.y_test)
        return OptimizationResult(
            str(destination), test["accuracy"], destination.stat().st_size / MB,
            test["latency_mean_ms"], None, selected["name"],
            {"validation_candidates": results, "test_samples": test["samples"],
             "latency_runs": test["latency_runs"],
             "selection_basis": "Smallest file first" if target.compute_capability == "tiny" else "Highest validation accuracy first",
             "latency_scope": "Local single-thread CPU TFLite, including I/O handling; not target-board latency"},
        )


class TinyMLOptimizer(EdgeOptimizer):
    def candidates(self, model):
        """Combine pretrained lightweight architectures with full INT8 quantization to reduce storage."""
        candidates = []
        for name in ("architecture_optimized", "nas_best"):
            path = ROOT / "part3/part3_results/edge_optimized_models" / f"{name}.keras"
            candidate = tf.keras.models.load_model(path, compile=False)
            candidates.append((f"{name}_int8", candidate, "int8"))
        return candidates


class MultiScaleDeploymentPipeline:
    def __init__(self):
        data = load_data()
        self.optimizers = {"cloud": CloudOptimizer(), "edge": EdgeOptimizer(data),
                           "tiny": TinyMLOptimizer(data)}
        # Each target sets file-size, latency, memory, and power limits.
        self.targets = {
            "cloud_server": DeploymentTarget("cloud_server", 1000.0, 100.0, 8000.0, 50000.0, "cloud"),
            "edge_device": DeploymentTarget("edge_device", 50.0, 200.0, 512.0, 2000.0, "edge"),
            "microcontroller": DeploymentTarget("microcontroller", 1.0, 1000.0, 64.0, 10.0, "tiny"),
        }

    def optimize_for_all_targets(self, baseline_model_path: str) -> Dict[str, OptimizationResult]:
        """Load the baseline once and dispatch each target to its optimizer."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        baseline = tf.keras.models.load_model(baseline_model_path, compile=False)
        results = {}
        for name, target in self.targets.items():
            print(f"Starting optimization: {name}", flush=True)
            results[name] = self.optimizers[target.compute_capability].optimize(baseline, target)
            result = results[name]
            print(f"Completed: {name}, accuracy {result.accuracy:.2%}, "
                  f"size {result.model_size_mb:.4f} MiB, reference latency {result.estimated_latency_ms:.3f} ms")
        return results

    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """Compare storage and accuracy while recording measurement scopes across runtimes."""
        baseline = load_json(ROOT / "part3/part3_results/edge_optimization_report.json")["baseline"]["tflite"]
        cloud = load_json(ROOT / "part2/part2_results/cloud_optimization_report.json")["results"]
        frontier = []
        for name, result in results.items():
            # Dominance requires no worse performance on either metric
            # and strictly better accuracy or file size.
            dominated = any(
                other.accuracy >= result.accuracy and other.model_size_mb <= result.model_size_mb
                and (other.accuracy > result.accuracy or other.model_size_mb < result.model_size_mb)
                for other in results.values())
            if not dominated:
                frontier.append(name)
        checks, comparisons = {}, {}
        for name, result in results.items():
            target = self.targets[name]
            checks[name] = {
                "model_storage_met": result.model_size_mb <= target.max_model_size_mb,
                "local_reference_latency_met": result.estimated_latency_ms <= target.max_latency_ms,
                "target_latency_met": None, "runtime_memory_met": None, "power_budget_met": None,
                "overall_status": "Partially verified; target-device feasibility remains unconfirmed",
            }
            comparisons[name] = {
                "accuracy_change_percentage_points": (result.accuracy - baseline["accuracy"]) * 100,
                "file_size_ratio_to_baseline_tflite": result.model_size_mb / baseline["size_mib"],
                "local_reference_latency_ms": result.estimated_latency_ms,
            }
        ratio = cloud["distributed_two"]["train_images_per_second"] / cloud["distributed_one"]["train_images_per_second"]
        return {
            "accuracy_size_pareto_frontier": frontier,
            "constraint_checks": checks, "baseline_comparisons": comparisons,
            "cloud_training_throughput_ratio_to_fp32": results["cloud_server"].details["train_images_per_second"] / cloud["fp32"]["train_images_per_second"],
            "simulated_multi_gpu_scaling": {"throughput_ratio": ratio, "two_replica_efficiency": ratio / 2,
                                            "scope": "Two logical replicas on one physical GPU; not physical multi-GPU scaling efficiency"},
            "bottlenecks": {"cloud_server": "GPU memory for large batches and synchronization overhead on physical multi-GPU systems",
                            "edge_device": "Quantization accuracy and operator implementations on the target CPU",
                            "microcontroller": "Activation memory, integer operator support, power consumption, and accuracy loss from a smaller architecture"},
            "comparison_scope": "Pareto analysis compares accuracy and deployment file size only. Cloud Keras and edge TFLite formats differ; GPU and CPU latency cannot establish cross-device speedup.",
        }

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Summarize constraint checks and deployment options."""
        recommendations = []
        for name, check in analysis["constraint_checks"].items():
            status = "meets" if check["model_storage_met"] and check["local_reference_latency_met"] else "does not meet"
            recommendations.append(f"{name}: {status} file-size and local reference latency limits; target-device latency, runtime memory, and power still require measurement.")
        recommendations.extend([
            "Use the selected high-throughput training configuration in the cloud; the distributed results only verify synchronization.",
            "Use the selected quantized model for offline edge inference, or a lightweight INT8 model under tighter resource limits.",
            "Cascade: process simple inputs on the microcontroller and forward low-confidence inputs to an edge device or the cloud; select thresholds on validation data.",
            "Hybrid deployment: retain a local model for offline use, and use the cloud for difficult inputs and model updates when connected.",
            "Development priorities: verify target-board operator support and activation memory, measure latency and power, then adjust accuracy-resource trade-offs.",
        ])
        return recommendations


def run_multi_scale_optimization():
    """Optimize each target, analyze trade-offs, generate recommendations, and save the JSON report."""
    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    pipeline = MultiScaleDeploymentPipeline()
    results = pipeline.optimize_for_all_targets(str(ROOT / "part1/baseline_model.keras"))
    analysis = pipeline.analyze_scaling_trade_offs(results)
    report = {
        "deployment_targets": {name: asdict(target) for name, target in pipeline.targets.items()},
        "optimization_results": {name: asdict(result) for name, result in results.items()},
        "scaling_analysis": analysis,
        "deployment_recommendations": pipeline.generate_deployment_recommendations(analysis),
        "measurement_scope": "Sizes use MiB. Cloud results reuse previous experiments; edge and TinyML models are quantized and evaluated in this run. null means unmeasured, not zero.",
    }
    path = OUTPUT / "multi_scale_optimization_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Multi-scale optimization complete. Report saved to: {path}")
    return report


if __name__ == "__main__":
    run_multi_scale_optimization()
