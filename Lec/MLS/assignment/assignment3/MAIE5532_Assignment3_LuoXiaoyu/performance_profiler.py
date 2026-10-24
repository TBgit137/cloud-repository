#!/usr/bin/env python3
"""第 4 部分：性能对比、Pareto 分析和硬件利用率建模。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
PART2_JSON = (
    PROJECT_ROOT / "part2" / "part2_results" / "performance_comparison.json"
)
PART3_RESULTS = PROJECT_ROOT / "part3" / "part3_results"
MODEL_PROFILES_JSON = PART3_RESULTS / "model_profiles.json"
ANALYTICAL_JSON = PART3_RESULTS / "analytical_estimates.json"
CROSS_PLATFORM_JSON = PART3_RESULTS / "cross_platform_analysis.json"
QEMU_JSON = PART3_RESULTS / "qemu_cortex_m.json"
WEBGPU_JSON = PART3_RESULTS / "webgpu_proxy.json"

MODEL_ORDER = [
    "baseline",
    "latency_optimized",
    "memory_optimized",
    "energy_optimized",
    "ptq_int8",
    "qat_int8",
    "mixed_precision",
    "dynamic_range",
]
PLATFORM_ORDER = [
    "x86_desktop",
    "arm_cortex_a78",
    "arm_cortex_m7",
    "webgpu_proxy",
]

PLATFORM_SPECS = {
    "x86_desktop": {
        "label": "[native-x86]",
        "effective_compute_gops": 75.0,
        "effective_bandwidth_gbps": 30.0,
        "thermal_derate": 0.95,
        "l1_cache_kb": 32,
        "l2_cache_kb": 256,
        "l3_cache_mb": 8,
        "memory_capacity_bytes": 8 * 1024 ** 3,
    },
    "arm_cortex_a78": {
        "label": "[modeled-ARM]",
        "effective_compute_gops": 10.0,
        "effective_bandwidth_gbps": 9.0,
        "thermal_derate": 0.8,
        "l1_cache_kb": 64,
        "l2_cache_kb": 512,
        "l3_cache_mb": 0,
        "memory_capacity_bytes": 512 * 1024 ** 2,
    },
    "arm_cortex_m7": {
        "label": "[QEMU-CM]",
        "effective_compute_gops": 0.45,
        "effective_bandwidth_gbps": 0.08,
        "thermal_derate": 0.65,
        "l1_cache_kb": 0,
        "l2_cache_kb": 0,
        "l3_cache_mb": 0,
        "memory_capacity_bytes": 512 * 1024,
    },
    "webgpu_proxy": {
        "label": "[WebGPU-proxy]",
        "effective_compute_gops": 1000.0,
        "effective_bandwidth_gbps": 100.0,
        "thermal_derate": 0.8,
        "l1_cache_kb": 0,
        "l2_cache_kb": 0,
        "l3_cache_mb": 0,
        "memory_capacity_bytes": 2 * 1024 ** 3,
    },
}


def _load_json(path: Path) -> dict:
    """读取 JSON，并在输入文件缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 Part 4 输入文件：{path}。请先完成对应的前序部分。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_part4_inputs() -> dict[str, dict]:
    """加载 Part 2 和 Part 3 的分析结果。"""
    return {
        "part2": _load_json(PART2_JSON),
        "profiles": _load_json(MODEL_PROFILES_JSON),
        "analytical": _load_json(ANALYTICAL_JSON),
        "cross_platform": _load_json(CROSS_PLATFORM_JSON),
        "qemu": _load_json(QEMU_JSON),
        "webgpu": _load_json(WEBGPU_JSON),
    }


def _dominates(left: dict, right: dict, objectives: dict[str, str]) -> bool:
    """判断 left 是否在所有目标上不差，且至少一个目标更优。"""
    strictly_better = False
    for metric, direction in objectives.items():
        left_value = float(left[metric])
        right_value = float(right[metric])
        tolerance = 1e-12 * max(1.0, abs(left_value), abs(right_value))
        if direction == "max":
            if left_value < right_value - tolerance:
                return False
            if left_value > right_value + tolerance:
                strictly_better = True
        elif direction == "min":
            if left_value > right_value + tolerance:
                return False
            if left_value < right_value - tolerance:
                strictly_better = True
        else:
            raise ValueError(f"未知优化方向：{direction}")
    return strictly_better


def pareto_frontier(
    rows: Iterable[dict],
    objectives: dict[str, str],
) -> list[dict]:
    """返回非支配解，不预设某个模型一定胜出。"""
    candidates = list(rows)
    frontier = []
    for candidate in candidates:
        if not any(
            _dominates(other, candidate, objectives)
            for other in candidates
            if other is not candidate
        ):
            frontier.append(candidate)
    return frontier


def _measured_rows(inputs: dict) -> list[dict]:
    """合并实测延迟、准确率和模型体积，并附上能耗估算来源。"""
    profiles = inputs["profiles"]
    analytical = inputs["analytical"]["x86_desktop"]
    part2 = inputs["part2"]
    rows = []
    for model in MODEL_ORDER:
        profile = profiles[model]
        energy_mj = float(analytical[model]["energy_j"]) * 1000.0
        metadata: dict[str, Any] = {}
        if model in part2.get("architecture_models", {}):
            source = part2["architecture_models"][model]
            for key in (
                "kernel_sparsity",
                "early_exit_samples",
                "early_exit_fraction",
            ):
                if key in source:
                    metadata[key] = source[key]
        if model in part2.get("quantized_models", {}):
            source = part2["quantized_models"][model]
            metadata["source_model"] = source.get("source_model")
            metadata["input_range"] = source.get("input_range")

        latency_ms = float(profile["measured_latency_ms"])
        rows.append(
            {
                "model": model,
                "accuracy": float(profile["accuracy"]),
                "latency_ms": latency_ms,
                "model_size_mb": float(profile["file_size_mb"]),
                "estimated_energy_mj": energy_mj,
                "throughput_fps": 1000.0 / latency_ms,
                "parameters": int(profile["parameters"]),
                "flops": int(profile["flops"]),
                "backend": profile["backend"],
                "latency_status": "measured on local x86 host",
                "energy_status": "analytical estimate",
                **metadata,
            }
        )
    return rows


def _platform_rows(
    inputs: dict,
    measured_rows: list[dict],
) -> list[dict]:
    """展开各平台的解析结果，并保留实测/估算状态。"""
    measured_by_model = {row["model"]: row for row in measured_rows}
    rows = []
    for platform in PLATFORM_ORDER:
        baseline = inputs["analytical"][platform]["baseline"]
        for model in MODEL_ORDER:
            metrics = inputs["analytical"][platform][model]
            measured = measured_by_model[model]
            rows.append(
                {
                    "platform": platform,
                    "execution_label": PLATFORM_SPECS[platform]["label"],
                    "model": model,
                    "accuracy": float(metrics["accuracy"]),
                    "estimated_latency_ms": float(metrics["latency_ms"]),
                    "estimated_memory_mb": (
                        float(metrics["memory_bytes"]) / (1024 ** 2)
                    ),
                    "estimated_energy_mj": (
                        float(metrics["energy_j"]) * 1000.0
                    ),
                    "fits_workspace": bool(metrics["fits_memory"]),
                    "estimated_speedup_vs_baseline": (
                        float(baseline["latency_ms"])
                        / float(metrics["latency_ms"])
                    ),
                    "model_size_mb": float(measured["model_size_mb"]),
                    "latency_status": "analytical estimate",
                    "energy_status": "analytical estimate",
                }
            )
    return rows


def generate_performance_report() -> dict:
    """生成第 4 部分所需的综合性能分析数据。"""
    performance_data = {
        "models": list(MODEL_ORDER),
        "platforms": list(PLATFORM_ORDER),
        "metrics": [
            "latency_ms",
            "memory_mb",
            "energy_mj",
            "accuracy",
            "throughput_fps",
        ],
    }
    inputs = load_part4_inputs()
    measured = _measured_rows(inputs)
    platforms = _platform_rows(inputs, measured)

    pareto_objectives = {
        "accuracy": "max",
        "latency_ms": "min",
        "model_size_mb": "min",
    }
    frontier = pareto_frontier(measured, pareto_objectives)
    platform_pareto = {}
    for platform in PLATFORM_ORDER:
        rows = [
            {
                **row,
                "latency_ms": row["estimated_latency_ms"],
                "energy_mj": row["estimated_energy_mj"],
                "memory_mb": row["estimated_memory_mb"],
            }
            for row in platforms
            if row["platform"] == platform
        ]
        platform_frontier = pareto_frontier(
            rows,
            {
                "accuracy": "max",
                "latency_ms": "min",
                "energy_mj": "min",
                "memory_mb": "min",
            },
        )
        platform_pareto[platform] = [
            row["model"] for row in platform_frontier
        ]

    batch_scaling = []
    for row in inputs["part2"]["memory_management"]["optimal_batch_size"][
        "candidates"
    ]:
        batch_scaling.append(
            {
                **row,
                "latency_per_sample_ms": (
                    float(row["latency_ms"]) / int(row["batch_size"])
                ),
            }
        )

    performance_data.update(
        {
            "measured": measured,
            "platform_estimates": platforms,
            "pareto": {
                "accuracy_latency_size": [
                    row["model"] for row in frontier
                ],
                "accuracy_latency": [
                    row["model"]
                    for row in pareto_frontier(
                        measured,
                        {
                            "accuracy": "max",
                            "latency_ms": "min",
                        },
                    )
                ],
                "accuracy_model_size": [
                    row["model"]
                    for row in pareto_frontier(
                        measured,
                        {
                            "accuracy": "max",
                            "model_size_mb": "min",
                        },
                    )
                ],
                "platforms": platform_pareto,
            },
            "batch_scaling": batch_scaling,
            "summary": {
                "best_accuracy_model": max(
                    measured,
                    key=lambda row: row["accuracy"],
                )["model"],
                "lowest_latency_model": min(
                    measured,
                    key=lambda row: row["latency_ms"],
                )["model"],
                "smallest_model": min(
                    measured,
                    key=lambda row: row["model_size_mb"],
                )["model"],
                "highest_batch_throughput": max(
                    batch_scaling,
                    key=lambda row: row["throughput_samples_per_second"],
                )["batch_size"],
            },
        }
    )
    return performance_data


def _cache_residency_proxy(
    working_set_bytes: int,
    platform: str,
) -> dict:
    """用工作集与 cache 容量的比例作为容量代理，不声称是实测命中率。"""
    specs = PLATFORM_SPECS[platform]
    result = {
        "status": "capacity proxy, not a measured hit rate",
        "platform": platform,
    }
    for cache_name in ("l1", "l2", "l3"):
        capacity_kb = specs.get(f"{cache_name}_cache_kb", 0)
        if not capacity_kb:
            result[f"{cache_name}_capacity_fraction"] = None
            continue
        capacity_bytes = int(capacity_kb * 1024)
        result[f"{cache_name}_capacity_fraction"] = min(
            1.0,
            capacity_bytes / working_set_bytes,
        )
    return result


def analyze_hardware_utilization(
    performance_data: dict | None = None,
) -> dict:
    """估算各模型在目标硬件上的计算、带宽、缓存代理和能效表现。"""
    inputs = load_part4_inputs()
    performance_data = performance_data or generate_performance_report()
    measured_by_model = {
        row["model"]: row for row in performance_data["measured"]
    }
    utilization_analysis = {
        "cpu_simd_utilization": {},
        "memory_bandwidth_efficiency": {},
        "cache_efficiency": {},
        "thermal_characteristics": {},
        "power_efficiency": {},
    }

    for platform_row in performance_data["platform_estimates"]:
        platform = platform_row["platform"]
        model = platform_row["model"]
        key = f"{platform}/{model}"
        specs = PLATFORM_SPECS[platform]
        profile = inputs["profiles"][model]
        latency_ms = float(platform_row["estimated_latency_ms"])
        memory_bytes = int(
            inputs["analytical"][platform][model]["memory_bytes"]
        )
        flops = int(profile["flops"])
        achieved_compute_gops = flops / (latency_ms * 1e6)
        achieved_bandwidth_gbps = memory_bytes / (latency_ms * 1e6)
        compute_utilization = (
            achieved_compute_gops / specs["effective_compute_gops"]
        )
        bandwidth_utilization = (
            achieved_bandwidth_gbps / specs["effective_bandwidth_gbps"]
        )
        compute_time_ms = (
            flops / (specs["effective_compute_gops"] * 1e9) * 1000.0
        )
        memory_time_ms = (
            memory_bytes
            / (specs["effective_bandwidth_gbps"] * 1e9)
            * 1000.0
        )
        energy_j = float(
            inputs["analytical"][platform][model]["energy_j"]
        )
        energy_pj_per_flop = energy_j * 1e12 / max(1, flops)
        measured = measured_by_model[model]

        utilization_analysis["cpu_simd_utilization"][key] = {
            "estimated_compute_utilization": compute_utilization,
            "achieved_gops": achieved_compute_gops,
            "effective_compute_gops": specs["effective_compute_gops"],
            "bottleneck": (
                "compute" if compute_time_ms >= memory_time_ms else "memory"
            ),
            "status": "analytical estimate",
        }
        utilization_analysis["memory_bandwidth_efficiency"][key] = {
            "memory_bandwidth_utilization": bandwidth_utilization,
            "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
            "effective_bandwidth_gbps": specs["effective_bandwidth_gbps"],
            "working_set_mb": memory_bytes / (1024 ** 2),
            "status": "analytical estimate",
        }
        utilization_analysis["cache_efficiency"][key] = (
            _cache_residency_proxy(memory_bytes, platform)
        )
        utilization_analysis["thermal_characteristics"][key] = {
            "thermal_derate": specs["thermal_derate"],
            "status": "assumption only; temperature was not measured",
        }
        utilization_analysis["power_efficiency"][key] = {
            "estimated_energy_mj": energy_j * 1000.0,
            "estimated_energy_pj_per_flop": energy_pj_per_flop,
            "measured_latency_ms": measured["latency_ms"],
            "status": "analytical estimate from operation counts",
        }

    qemu = inputs["qemu"]
    utilization_analysis["mcu_memory_validity"] = {
        "model_bytes": int(qemu["model_bytes"]),
        "tensor_arena_used_bytes": int(qemu["tensor_arena_used_bytes"]),
        "flash_limit_bytes": 2 * 1024 * 1024,
        "sram_limit_bytes": 512 * 1024,
        "fits_flash": int(qemu["model_bytes"]) <= 2 * 1024 * 1024,
        "fits_sram": (
            int(qemu["tensor_arena_used_bytes"]) <= 512 * 1024
        ),
        "status": "TFLite Micro arena measured inside QEMU-CM",
    }
    utilization_analysis["webgpu_proxy"] = {
        "mean_ms": float(inputs["webgpu"]["mean_ms"]),
        "median_ms": float(inputs["webgpu"]["median_ms"]),
        "p95_ms": float(inputs["webgpu"]["p95_ms"]),
        "status": "desktop proxy; not a smartphone measurement",
    }
    return utilization_analysis
