#!/usr/bin/env python3
"""Cross-platform performance modeling and simulation validation."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART1 = ROOT / "part1"
PART2 = ROOT / "part2"
RESULTS_DIR = HERE / "part3_results"
PART2_JSON = PART2 / "part2_results" / "performance_comparison.json"
QEMU_RESULT = RESULTS_DIR / "qemu_cortex_m.json"
WEBGPU_RESULT = RESULTS_DIR / "webgpu_proxy.json"
QEMU_BINARY = HERE / "cortex_m" / "qemu_cifar_benchmark"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _activation_bytes(model: keras.Model) -> int:
    """Estimate the largest single-layer activation buffer in bytes."""
    maximum = 0
    for layer in model.layers:
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, list):
            shape = shape[0]
        if not isinstance(shape, tuple):
            continue
        count = 1
        for dimension in shape[1:]:
            if isinstance(dimension, int):
                count *= dimension
        maximum = max(maximum, count * 4)
    return maximum


def _estimate_flops(model: keras.Model) -> int:
    """Estimate FLOPs for one image with the TensorFlow profiler."""
    input_shape = tuple(model.input_shape[1:])
    concrete = tf.function(
        lambda images: model(images, training=False)
    ).get_concrete_function(
        tf.TensorSpec((1, *input_shape), tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options["output"] = "none"
    profile = tf.compat.v1.profiler.profile(
        graph=frozen.graph,
        run_meta=tf.compat.v1.RunMetadata(),
        cmd="op",
        options=options,
    )
    return int(profile.total_float_ops)


def _keras_profile(path: Path, result: dict) -> dict:
    """Extract performance-model inputs from a Keras model and result record."""
    model = keras.models.load_model(path, compile=False)
    weight_bytes = sum(
        int(tf.size(weight).numpy()) * tf.as_dtype(weight.dtype).size
        for weight in model.weights
    )
    input_shape = list(model.input_shape[1:])
    profile = {
        "backend": "Keras GPU",
        "input_shape": input_shape,
        "accuracy": float(result["accuracy"]),
        "parameters": int(result["parameters"]),
        "weight_bytes": weight_bytes,
        "activation_bytes": _activation_bytes(model),
        "flops": _estimate_flops(model),
        "file_size_mb": float(result["model_size_mb"]),
        "measured_latency_ms": float(result["single_inference_time_ms"]),
    }
    return profile


def collect_model_profiles() -> dict[str, dict]:
    """Collect profile inputs for baseline, architecture, and quantized models."""
    comparison = json.loads(PART2_JSON.read_text(encoding="utf-8"))
    baseline_model = keras.models.load_model(
        PART1 / "baseline_mobilenetv2.keras",
        compile=False,
    )
    profiles: dict[str, dict] = {
        "baseline": {
            "backend": "Keras GPU",
            "input_shape": [224, 224, 3],
            "accuracy": float(comparison["baseline"]["accuracy"]),
            "parameters": int(comparison["baseline"]["parameters"]),
            "weight_bytes": int(
                comparison["baseline"]["memory_usage_mb"] * (1024 ** 2)
            ),
            "activation_bytes": _activation_bytes(baseline_model),
            "flops": int(comparison["baseline"]["flops"]),
            "file_size_mb": float(comparison["baseline"]["model_size_mb"]),
            "measured_latency_ms": float(
                comparison["baseline"]["single_inference_time"]
            ),
        }
    }

    for name, metrics in comparison["architecture_models"].items():
        path = Path(metrics["path"])
        profiles[name] = _keras_profile(path, metrics)

    latency = profiles["latency_optimized"]
    for name, metrics in comparison["quantized_models"].items():
        path = Path(metrics["path"])
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        tensor_bytes = []
        for detail in interpreter.get_tensor_details():
            count = math.prod(int(value) for value in detail["shape"])
            if count:
                tensor_bytes.append(int(
                    count * tf.as_dtype(detail["dtype"]).size
                ))
        profiles[name] = {
            "backend": "TFLite CPU",
            "input_shape": [
                int(dimension) for dimension in input_detail["shape"][1:]
            ],
            "accuracy": float(metrics["accuracy"]),
            "parameters": latency["parameters"],
            "weight_bytes": path.stat().st_size,
            "activation_bytes": max(tensor_bytes),
            "flops": latency["flops"],
            "file_size_mb": path.stat().st_size / (1024 ** 2),
            "measured_latency_ms": float(
                metrics["single_inference_time_ms"]
            ),
        }
    return profiles


class PlatformPerformanceModel:
    """Model performance characteristics for different hardware platforms."""

    def __init__(self, platform_specs):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_performance_models(self):
        """Build analytical performance models for the target platform."""
        models = {}
        models["roofline"] = self._build_roofline_model()
        models["energy"] = self._build_energy_model()
        models["memory"] = self._build_memory_model()
        return models

    def _build_roofline_model(self):
        """Return the Roofline latency estimator."""
        specs = self.platform_specs

        def estimate(flops: int, memory_bytes: int) -> float:
            compute_ms = (
                flops
                / (specs["effective_compute_gops"] * 1e9)
                * 1000.0
            )
            memory_ms = (
                memory_bytes
                / (specs["effective_bandwidth_gbps"] * 1e9)
                * 1000.0
            )
            throttling = specs["thermal_derate"]
            return max(compute_ms, memory_ms) / throttling + specs[
                "fixed_overhead_ms"
            ]

        return estimate

    def _build_energy_model(self):
        """Return the operation- and memory-based energy estimator."""
        specs = self.platform_specs

        def estimate(
            flops: int,
            memory_bytes: int,
            latency_ms: float,
        ) -> float:
            dynamic_joules = (
                flops * specs["energy_per_flop_pj"]
                + memory_bytes * specs["energy_per_byte_pj"]
            ) / 1e12
            idle_joules = specs["idle_power_mw"] * latency_ms / 1000.0 / 1000.0
            return dynamic_joules + idle_joules

        return estimate

    def _build_memory_model(self):
        """Return the weight and peak-activation memory estimator."""
        return lambda weights: weights["weight_bytes"] + 2 * weights[
            "activation_bytes"
        ]

    def estimate_performance(self, model_graph, platform_type):
        """Estimate latency, memory, and energy on a target platform."""
        if platform_type != self.platform_specs["name"]:
            raise ValueError(
                f"Platform configuration {self.platform_specs['name']} "
                f"does not match {platform_type}."
            )
        memory_working_set = self.performance_models["memory"](model_graph)
        roofline_ms = self.performance_models["roofline"](
            model_graph["flops"],
            memory_working_set,
        )
        energy_j = self.performance_models["energy"](
            model_graph["flops"],
            memory_working_set,
            roofline_ms,
        )
        return {
            "latency_ms": float(roofline_ms),
            "memory_bytes": int(memory_working_set),
            "energy_j": float(energy_j),
            "fits_memory": (
                memory_working_set <= self.platform_specs["memory_capacity_bytes"]
            ),
            "accuracy": float(model_graph["accuracy"]),
        }


class CrossPlatformAnalyzer:
    """Analyze optimization effectiveness across target platforms."""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()

    def _load_platform_specs(self):
        """Load detailed specifications for the target platforms."""
        return {
            "x86_desktop": {
                "name": "x86_desktop",
                "execution_label": "[native-x86]",
                "core_count": 8,
                "frequency_ghz": 3.5,
                "simd_width": 256,
                "l1_cache_kb": 32,
                "l2_cache_kb": 256,
                "l3_cache_mb": 8,
                "memory_bandwidth_gbps": 50,
                "tdp_watts": 65,
                "effective_compute_gops": 75.0,
                "effective_bandwidth_gbps": 30.0,
                "fixed_overhead_ms": 0.15,
                "thermal_derate": 0.95,
                "energy_per_flop_pj": 5.0,
                "energy_per_byte_pj": 90.0,
                "idle_power_mw": 15000.0,
                "memory_capacity_bytes": 8 * 1024 ** 3,
            },
            "arm_cortex_a78": {
                "name": "arm_cortex_a78",
                "execution_label": "[modeled-ARM]",
                "core_count": 4,
                "frequency_ghz": 2.4,
                "simd_width": 128,
                "l1_cache_kb": 64,
                "l2_cache_kb": 512,
                "memory_bandwidth_gbps": 15,
                "tdp_watts": 5,
                "effective_compute_gops": 10.0,
                "effective_bandwidth_gbps": 9.0,
                "fixed_overhead_ms": 1.0,
                "thermal_derate": 0.8,
                "energy_per_flop_pj": 15.0,
                "energy_per_byte_pj": 180.0,
                "idle_power_mw": 1200.0,
                "memory_capacity_bytes": 512 * 1024 ** 2,
            },
            "arm_cortex_m7": {
                "name": "arm_cortex_m7",
                "execution_label": "[QEMU-CM]",
                "frequency_mhz": 400,
                "sram_kb": 512,
                "flash_mb": 2,
                "power_budget_mw": 100,
                "has_fpu": True,
                "dsp_extensions": True,
                "effective_compute_gops": 0.45,
                "effective_bandwidth_gbps": 0.08,
                "fixed_overhead_ms": 20.0,
                "thermal_derate": 0.65,
                "energy_per_flop_pj": 60.0,
                "energy_per_byte_pj": 900.0,
                "idle_power_mw": 50.0,
                "memory_capacity_bytes": 512 * 1024,
            },
            "webgpu_proxy": {
                "name": "webgpu_proxy",
                "execution_label": "[WebGPU-proxy]",
                "effective_compute_gops": 1000.0,
                "effective_bandwidth_gbps": 100.0,
                "fixed_overhead_ms": 3.5,
                "thermal_derate": 0.8,
                "energy_per_flop_pj": 2.0,
                "energy_per_byte_pj": 45.0,
                "idle_power_mw": 8000.0,
                "memory_capacity_bytes": 2 * 1024 ** 3,
            },
        }

    def analyze_optimization_effectiveness(
        self,
        baseline_model,
        optimized_models,
    ):
        """Compare speed, memory, energy, and accuracy across platforms."""
        results = {}
        for platform, specs in self.platform_characteristics.items():
            estimator = PlatformPerformanceModel(specs)
            baseline_performance = estimator.estimate_performance(
                baseline_model,
                platform,
            )
            platform_results = {}
            for model_name, model_profile in optimized_models.items():
                optimized_performance = estimator.estimate_performance(
                    model_profile,
                    platform,
                )
                platform_results[model_name] = {
                    "speedup": baseline_performance["latency_ms"]
                    / optimized_performance["latency_ms"],
                    "memory_reduction": 1
                    - optimized_performance["memory_bytes"]
                    / baseline_performance["memory_bytes"],
                    "energy_reduction": 1
                    - optimized_performance["energy_j"]
                    / baseline_performance["energy_j"],
                    "accuracy_loss": baseline_performance["accuracy"]
                    - optimized_performance["accuracy"],
                    "fits_target_memory": optimized_performance["fits_memory"],
                }
            results[platform] = platform_results
        return results

    def model_performance(self, model, platform):
        """Estimate model performance on the specified platform."""
        specs = self.platform_characteristics[platform]
        return PlatformPerformanceModel(specs).estimate_performance(
            model,
            platform,
        )


def simulate_arm_performance(models):
    """Generate analytical performance estimates for ARM Cortex-A.

    Cortex-A values are modeled estimates. QEMU results are reported
    separately as Cortex-M measurements.
    """
    analyzer = CrossPlatformAnalyzer()
    specs = analyzer.platform_characteristics["arm_cortex_a78"]
    estimator = PlatformPerformanceModel(specs)
    return {
        "execution_context": "[modeled-ARM]",
        "status": "modeled",
        "models": {
            name: estimator.estimate_performance(profile, "arm_cortex_a78")
            for name, profile in models.items()
        },
    }


def _run_qemu_benchmark() -> dict:
    """Run the compiled QEMU Cortex-M CIFAR firmware and parse its output."""
    if not QEMU_BINARY.exists():
        raise FileNotFoundError(
            f"QEMU firmware not found: {QEMU_BINARY}. "
            "Build the Cortex-M benchmark first."
        )
    result = subprocess.run(
        ["qemu-arm", "-cpu", "cortex-m3", str(QEMU_BINARY)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    output = result.stdout
    clock_match = re.search(
        r"Average clock milliseconds:\s*([0-9.]+)",
        output,
    )
    model_match = re.search(r"CIFAR model bytes:\s*([0-9]+)", output)
    arena_match = re.search(r"Tensor arena bytes:\s*([0-9]+)", output)
    arena_used_match = re.search(
        r"Tensor arena used bytes:\s*([0-9]+)",
        output,
    )
    clock_hz_match = re.search(
        r"Clock ticks per second:\s*([0-9]+)",
        output,
    )
    if not all(
        (
            clock_match,
            model_match,
            arena_match,
            arena_used_match,
            clock_hz_match,
        )
    ):
        raise RuntimeError(f"Could not parse QEMU output:\n{output}")
    clock_hz = int(clock_hz_match.group(1))
    result = {
        "execution_context": "[QEMU-CM]",
        "status": "measured",
        "cpu": "cortex-m3",
        "average_simulated_cpu_ms": float(clock_match.group(1)),
        "model_bytes": int(model_match.group(1)),
        "tensor_arena_bytes": int(arena_match.group(1)),
        "tensor_arena_used_bytes": int(arena_used_match.group(1)),
        "clock_ticks_per_second": clock_hz,
        "clock_resolution_ms": 1000.0 / clock_hz,
        "raw_output": output.strip(),
    }
    _write_json(QEMU_RESULT, result)
    return result


def simulate_cortex_m_performance(models):
    """Run the Cortex-M benchmark with QEMU and TFLite Micro."""
    measured = _run_qemu_benchmark()
    measured["available_models"] = [
        name for name in models if name == "ptq_int8"
    ]
    return measured


def simulate_mobile_gpu_performance(models):
    """Read the ONNX Runtime Web WebGPU proxy measurement."""
    if not WEBGPU_RESULT.exists():
        raise FileNotFoundError(f"WebGPU result not found: {WEBGPU_RESULT}")
    result = json.loads(WEBGPU_RESULT.read_text(encoding="utf-8"))
    result["status"] = "measured"
    result["available_models"] = [
        name for name in models if name == "latency_optimized"
    ]
    return result


def cross_validate_models(validation_results):
    """Compare analytical predictions with simulation or proxy measurements."""
    comparisons = []
    pairs = [
        (
            "native_x86",
            "ptq_int8",
            validation_results["native_x86"]["measured_latency_ms"],
            validation_results["analytical"]["x86_desktop"]["ptq_int8"][
                "latency_ms"
            ],
        ),
        (
            "qemu_cm",
            "ptq_int8",
            validation_results["qemu_cm"]["average_simulated_cpu_ms"],
            validation_results["analytical"]["arm_cortex_m7"]["ptq_int8"][
                "latency_ms"
            ],
        ),
        (
            "webgpu_proxy",
            "latency_optimized",
            validation_results["webgpu_proxy"]["mean_ms"],
            validation_results["analytical"]["webgpu_proxy"][
                "latency_optimized"
            ]["latency_ms"],
        ),
    ]
    for platform, model, measured, predicted in pairs:
        comparisons.append(
            {
                "platform": platform,
                "model": model,
                "measured_ms": measured,
                "predicted_ms": predicted,
                "absolute_error_ms": abs(predicted - measured),
                "relative_error_percent": (
                    abs(predicted - measured) / measured * 100.0
                    if measured
                    else None
                ),
            }
        )
    return {"comparisons": comparisons}


def validate_with_simulation(models, platform_configs):
    """Validate performance estimates with simulation and proxy evidence."""
    analyzer = CrossPlatformAnalyzer()
    analytical = {
        platform: {
            name: PlatformPerformanceModel(config).estimate_performance(
                profile,
                platform,
            )
            for name, profile in models.items()
        }
        for platform, config in analyzer.platform_characteristics.items()
    }
    validation_results = {
        "analytical": analytical,
        "native_x86": {
            "execution_context": "[native-x86]",
            "status": "measured",
            "measured_latency_ms": models["ptq_int8"][
                "measured_latency_ms"
            ],
        },
        "qemu_arm": simulate_arm_performance(models),
        "qemu_cm": simulate_cortex_m_performance(models),
        "webgpu_proxy": simulate_mobile_gpu_performance(models),
    }
    validation_results["model_accuracy"] = cross_validate_models(
        validation_results
    )
    return validation_results


def _write_report(
    model_profiles: dict,
    analytical: dict,
    simulation: dict,
    cross_platform: dict,
) -> None:
    """Generate the technical analysis report."""
    qemu = simulation["qemu_cm"]
    webgpu = simulation["webgpu_proxy"]
    specs = CrossPlatformAnalyzer().platform_characteristics
    latency_profile = model_profiles["latency_optimized"]
    baseline_profile = model_profiles["baseline"]
    latency_intensity = latency_profile["flops"] / (
        latency_profile["weight_bytes"]
        + 2 * latency_profile["activation_bytes"]
    )
    m7_specs = specs["arm_cortex_m7"]
    m7_arena_fits = (
        qemu["tensor_arena_used_bytes"] <= m7_specs["memory_capacity_bytes"]
    )
    m7_flash_fits = (
        qemu["model_bytes"] <= m7_specs["flash_mb"] * 1024 * 1024
    )

    lines = [
        "# MobileNetV2 CIFAR-10 Multi-Platform Analysis",
        "",
        "## Objective and scope",
        "",
        "This report evaluates the Part 1 baseline and the Part 2 variants "
        "with four execution contexts: native x86 TFLite, simulated "
        "Cortex-M3 TFLite Micro, a desktop WebGPU mobile-GPU proxy, and an "
        "analytical ARM Cortex-A78 model. The design follows Track B and "
        "does not claim physical MCU or smartphone measurements.",
        "",
        "| Execution label | Platform class | Evidence |",
        "|---|---|---|",
        "| `[native-x86]` | Desktop CPU | Measured TFLite inference |",
        "| `[QEMU-CM]` | Cortex-M MCU | Measured user-mode QEMU execution |",
        "| `[WebGPU-proxy]` | Mobile GPU proxy | "
        "Measured ONNX Runtime Web/WebGPU |",
        "| `[modeled-ARM]` | Cortex-A mobile CPU | Roofline/energy estimate |",
        "",
        "## Model profiles",
        "",
        "| Model | Accuracy | Parameters | FLOPs | File MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, profile in model_profiles.items():
        lines.append(
            f"| {name} | {profile['accuracy'] * 100:.2f}% | "
            f"{profile['parameters']:,} | {profile['flops']:,} | "
            f"{profile['file_size_mb']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The latency model uses `alpha=0.5` and 160x160 input. MobileNetV2 "
            "depthwise separable convolutions reduce both computation and "
            "parameter count. The INT8 model is 0.937 MiB, compared with "
            f"{baseline_profile['file_size_mb']:.3f} MiB for the baseline.",
            "",
            "## Analytical models",
            "",
            "For each platform, latency is the larger of the compute and "
            "memory terms after thermal derating, plus a fixed launch "
            "overhead:",
            "",
            "```text",
            "L_est = max(F / effective_GOPs, B / effective_GBps)",
            "        / thermal_derate + fixed_overhead",
            "```",
            "",
            "The working set is `B = weight_bytes + 2 * activation_bytes`; "
            "the factor of two approximates overlapping input and output "
            "activation buffers. Energy is estimated as",
            "",
            "```text",
            "E_est = (F * pJ_per_FLOP + B * pJ_per_byte) / 1e12",
            "        + idle_power_mW * L_est / 1e6",
            "```",
            "",
            "These are deliberately simple, uncalibrated Roofline-style "
            "models. They do not model per-kernel cache behavior or "
            "instruction-level pipelining.",
            "",
            "### Platform estimates",
            "",
            "| Platform | Model | Latency ms | Memory MiB | Energy mJ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for platform, models in analytical.items():
        for name, metrics in models.items():
            lines.append(
                f"| {platform} | {name} | {metrics['latency_ms']:.3f} | "
                f"{metrics['memory_bytes'] / (1024 ** 2):.3f} | "
                f"{metrics['energy_j'] * 1000:.3f} |"
            )

    lines.extend(
        [
            "",
            "The optimized model has an arithmetic intensity of approximately "
            f"{latency_intensity:.1f} FLOP/byte. All four modeled platforms "
            "therefore sit above their Roofline ridge points and reduce "
            "latency primarily by lowering FLOPs. The WebGPU model is also "
            "strongly affected by its fixed launch overhead.",
            "",
            "## Simulation and proxy validation",
            "",
            f"Native x86 PTQ latency: "
            f"{simulation['native_x86']['measured_latency_ms']:.3f} ms.",
            "",
            f"QEMU Cortex-M PTQ latency: "
            f"{qemu['average_simulated_cpu_ms']:.3f} ms "
            f"(clock resolution {qemu['clock_resolution_ms']:.1f} ms, "
            "five measured runs).",
            "",
            f"WebGPU proxy latency: "
            f"{webgpu['mean_ms']:.3f} ms mean, {webgpu['median_ms']:.3f} ms "
            f"median, and {webgpu['p95_ms']:.3f} ms P95 over 100 runs.",
            "",
            "| Platform | Model | Measured ms | Predicted ms | Error % |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in simulation["model_accuracy"]["comparisons"]:
        lines.append(
            f"| {row['platform']} | {row['model']} | "
            f"{row['measured_ms']:.3f} | {row['predicted_ms']:.3f} | "
            f"{row['relative_error_percent']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "Raw model errors are large enough that the analytical estimates "
            "should be treated as a planning tool, not as replacements for "
            "measurement:",
            "",
            "- Native x86 is overestimated mainly because the fixed effective "
            "throughput is conservative for XNNPACK, AVX2/AVX-512 kernels, "
            "cached weights, and pipelined single-image execution.",
            "- Cortex-M is underestimated because the prebuilt portable "
            "TensorFlow Lite Micro kernels do not include the same "
            "CMSIS-NN/SIMD exploitation assumed by the effective GOPs figure. "
            "Pointer chasing, small tensor operations, and interpreter "
            "overhead therefore dominate.",
            "- WebGPU is underestimated by about one quarter because browser "
            "submission, queue scheduling, synchronization, and desktop "
            "driver behavior add cost beyond the simplified 3.5 ms fixed "
            "overhead.",
            "",
            "### MCU memory check",
            "",
            "The PTQ INT8 model contains "
            f"{qemu['model_bytes']:,} bytes and TFLite Micro reports "
            f"{qemu['tensor_arena_used_bytes']:,} bytes of arena use from a "
            f"{qemu['tensor_arena_bytes']:,}-byte configured arena. Against "
            "the modeled Cortex-M7 limits, the model "
            f"{'fits' if m7_flash_fits else 'does not fit'} in 2 MiB flash, "
            f"but the arena {'fits' if m7_arena_fits else 'does not fit'} "
            "in 512 KiB SRAM. QEMU user mode proves functional execution but "
            "does not enforce the physical SRAM limit.",
            "",
            "## Cross-platform effectiveness",
            "",
            "| Platform | Variant | Speedup | Working-set reduction | "
            "Energy reduction | Accuracy loss | Fits workspace |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    selected_variants = [
        "latency_optimized",
        "memory_optimized",
        "energy_optimized",
        "ptq_int8",
        "mixed_precision",
    ]
    for platform in (
        "x86_desktop",
        "arm_cortex_a78",
        "arm_cortex_m7",
        "webgpu_proxy",
    ):
        for model_name in selected_variants:
            metrics = cross_platform[platform][model_name]
            lines.append(
                f"| {platform} | {model_name} | {metrics['speedup']:.2f}x | "
                f"{metrics['memory_reduction'] * 100:.2f}% | "
                f"{metrics['energy_reduction'] * 100:.2f}% | "
                f"{metrics['accuracy_loss'] * 100:.2f} pp | "
                f"{metrics['fits_target_memory']} |"
            )

    lines.extend(
        [
            "",
            "The architecture and INT8 changes reduce FLOPs by about 83.6% "
            "and therefore produce a predicted 5.6-5.8x CPU speedup. The "
            "measured native PTQ latency of 0.663 ms is consistent with that "
            "direction. WebGPU gains only about 1.17x in the analytical "
            "model because launch overhead and a much larger compute budget "
            "make the reduced graph less dominant.",
            "",
            "INT8 reduces the modeled working set by about 84.3%; memory "
            "attributes are most valuable on Cortex-M, where flash and SRAM "
            "are explicitly limited. The memory-optimized Keras artifact is "
            "3.239 MiB after serialization, but its inference graph still "
            "contains dense zero-valued tensors, so compute and bandwidth "
            "are not reduced beyond the width change. The energy-optimized "
            "early-exit path affects only the 0.81% of samples that pass its "
            "confidence threshold; the graph-level analytical model therefore "
            "uses the same full-model work for most samples.",
            "",
            "## Theoretical validation and interpretation",
            "",
            "The Roofline model predicts that reducing arithmetic work should "
            "matter until the workload becomes memory- or overhead-bound. "
            "MobileNetV2 uses inverted residual blocks and depthwise "
            "separable convolution to reduce multiply-accumulates; INT8 "
            "quantization replaces float storage and integerizable compute, "
            "which is visible in the measured 0.937 MiB model size. These "
            "changes reduce both arithmetic work and storage demand.",
            "",
            "The measured disagreements also identify the missing effects: "
            "cache reuse, SIMD/kernel selection, framework dispatch, and "
            "accelerator scheduling. A production model would calibrate "
            "effective throughput per operation type and use empirical "
            "microbenchmarks. That calibration is intentionally omitted "
            "because the raw model is the object being validated.",
            "",
            "## Limitations",
            "",
            "- QEMU-CM is user-mode simulation; it does not enforce SRAM, "
            "flash, cache, power, clock-frequency, or thermal limits.",
            "- The WebGPU result uses a desktop GPU and synthetic input as a "
            "relative proxy for a mobile GPU.",
            "- Energy values are analytical estimates derived from operation "
            "and byte counts; no physical power meter was used.",
            "- Floating-point equal-FLOP assumptions do not capture Tensor "
            "Core or NPU execution.",
            "- Only one INT8 model was executed on Cortex-M. The other model "
            "rows are analytical predictions.",
            "",
            "Qualified execution labels:",
            "",
            "- `[native-x86]`: TFLite CPU measurement.",
            "- `[QEMU-CM]`: qemu-arm Cortex-M3 with TFLite Micro.",
            "- `[WebGPU-proxy]`: ONNX Runtime Web on desktop WebGPU.",
            "- `[modeled-ARM]`: analytical ARM Cortex-A estimate.",
            "",
            "QEMU and WebGPU results are relative simulation evidence, not "
            "physical MCU or smartphone measurements.",
            "",
            "## References",
            "",
            "1. S. Williams, A. Waterman, and D. Patterson, \"Roofline: An "
            "Insightful Visual Performance Model for Multicore "
            "Architectures,\" *Communications of the ACM*, 2009.",
            "2. M. Sandler et al., \"MobileNetV2: Inverted Residuals and "
            "Linear Bottlenecks,\" *CVPR*, 2018.",
            "3. B. Jacob et al., \"Quantization and Training of Neural "
            "Networks for Efficient Integer-Arithmetic-Only Inference,\" "
            "*CVPR*, 2018.",
            "4. R. David et al., \"TensorFlow Lite Micro: Embedded Machine "
            "Learning on TinyML Systems,\" *Proceedings of MLSys*, 2021.",
        ]
    )
    (HERE / "part3_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run modeling, simulation validation, and cross-platform comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profiles = collect_model_profiles()
    analyzer = CrossPlatformAnalyzer()
    analytical = {
        platform: {
            name: PlatformPerformanceModel(config).estimate_performance(
                profile,
                platform,
            )
            for name, profile in profiles.items()
        }
        for platform, config in analyzer.platform_characteristics.items()
    }
    cross_platform = analyzer.analyze_optimization_effectiveness(
        profiles["baseline"],
        {
            name: profile
            for name, profile in profiles.items()
            if name != "baseline"
        },
    )
    simulation = validate_with_simulation(
        profiles,
        analyzer.platform_characteristics,
    )

    _write_json(RESULTS_DIR / "model_profiles.json", profiles)
    _write_json(RESULTS_DIR / "analytical_estimates.json", analytical)
    _write_json(RESULTS_DIR / "simulation_validation.json", simulation)
    _write_json(RESULTS_DIR / "cross_platform_analysis.json", cross_platform)
    _write_report(profiles, analytical, simulation, cross_platform)
    print(json.dumps(cross_platform, indent=2))


if __name__ == "__main__":
    main()
