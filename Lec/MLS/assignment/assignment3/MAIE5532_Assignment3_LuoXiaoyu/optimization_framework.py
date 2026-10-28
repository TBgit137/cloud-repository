#!/usr/bin/env python3
"""Hardware constraints, optimization priorities, and design-space exploration."""

from __future__ import annotations

from typing import Any

from performance_profiler import (
    PLATFORM_ORDER,
    generate_performance_report,
)


class HardwareAwareDesignMethodology:
    """Framework for hardware-aware machine-learning system design."""

    def __init__(self):
        self.design_principles = [
            "co_design_hardware_software",
            "early_constraint_specification",
            "iterative_optimization",
            "multi_objective_optimization",
            "platform_specific_tuning",
        ]

    def constraint_specification_framework(
        self,
        application_requirements,
    ):
        """Convert application needs into checkable constraints before training."""
        requirements = dict(application_requirements)
        platform = requirements.get("platform", "x86_desktop")

        aliases = {
            "accuracy_target": (
                "accuracy_target",
                "accuracy",
                "minimum_accuracy",
            ),
            "max_latency_ms": (
                "max_latency_ms",
                "latency_ms",
                "latency_requirement_ms",
            ),
            "max_model_size_mb": (
                "max_model_size_mb",
                "model_size_mb",
                "memory_limit_mb",
            ),
            "max_energy_mj": (
                "max_energy_mj",
                "energy_mj",
                "power_budget_mj",
            ),
            "max_working_set_mb": (
                "max_working_set_mb",
                "working_set_mb",
                "sram_mb",
            ),
        }

        normalized: dict[str, Any] = {"platform": platform}
        missing = []
        for target_key, source_keys in aliases.items():
            value = next(
                (
                    requirements[key]
                    for key in source_keys
                    if key in requirements and requirements[key] is not None
                ),
                None,
            )
            normalized[target_key] = (
                None if value is None else float(value)
            )
            if value is None and target_key in {
                "accuracy_target",
                "max_latency_ms",
                "max_model_size_mb",
            }:
                missing.append(target_key)

        invalid = [
            key
            for key in (
                "accuracy_target",
                "max_latency_ms",
                "max_model_size_mb",
                "max_energy_mj",
                "max_working_set_mb",
            )
            if normalized[key] is not None and normalized[key] <= 0
        ]
        if platform not in PLATFORM_ORDER:
            invalid.append("platform")

        normalized.update(
            {
                "is_complete": not missing and not invalid,
                "missing_fields": missing,
                "invalid_fields": invalid,
                "specification_stage": "before model training",
            }
        )
        return normalized

    def optimization_priority_framework(self, constraints):
        """Return an ordered optimization path based on the tightest constraints."""
        normalized = (
            constraints
            if "is_complete" in constraints
            else self.constraint_specification_framework(constraints)
        )
        platform = normalized["platform"]
        latency = normalized["max_latency_ms"]
        model_size = normalized["max_model_size_mb"]
        energy = normalized["max_energy_mj"]
        accuracy = normalized["accuracy_target"]
        priorities = []

        if platform == "arm_cortex_m7" or (
            model_size is not None and model_size < 5.0
        ):
            priorities.extend(
                [
                    {
                        "priority": 1,
                        "technique": "INT8 post-training quantization",
                        "reason": "reduce flash, bandwidth and arithmetic cost",
                    },
                    {
                        "priority": 2,
                        "technique": "reduce input resolution and width",
                        "reason": "lower activation and peak working-set size",
                    },
                    {
                        "priority": 3,
                        "technique": "pruning or early exit",
                        "reason": "remove work when accuracy budget permits",
                    },
                ]
            )
        if latency is not None and latency < 10.0:
            priorities.append(
                {
                    "priority": len(priorities) + 1,
                    "technique": "latency-optimized architecture",
                    "reason": "reduce FLOPs before relying on kernel tuning",
                }
            )
        if energy is not None and energy < 40.0:
            priorities.append(
                {
                    "priority": len(priorities) + 1,
                    "technique": "energy-aware inference",
                    "reason": "combine INT8 with early exit when confidence "
                    "thresholds are reliable",
                }
            )
        if accuracy is not None and accuracy >= 0.90:
            priorities.append(
                {
                    "priority": len(priorities) + 1,
                    "technique": "mixed precision or reduced fine-tuning",
                    "reason": "retain higher accuracy after width and input "
                    "reduction",
                }
            )
        if not priorities:
            priorities.append(
                {
                    "priority": 1,
                    "technique": "profile before optimizing",
                    "reason": "no active performance constraint is tighter "
                    "than the current design",
                }
            )
        return {
            "platform": platform,
            "constraints_complete": normalized["is_complete"],
            "priorities": priorities,
        }

    def design_space_exploration(self, base_model, constraints):
        """Filter existing variants against constraints without retraining."""
        normalized = (
            constraints
            if "is_complete" in constraints
            else self.constraint_specification_framework(constraints)
        )
        performance = generate_performance_report()
        platform = normalized["platform"]
        candidates = []

        if platform == "x86_desktop":
            source_rows = performance["measured"]
        else:
            source_rows = [
                {
                    **row,
                    "latency_ms": row["estimated_latency_ms"],
                    "model_size_mb": row["model_size_mb"],
                    "energy_mj": row["estimated_energy_mj"],
                    "working_set_mb": row["estimated_memory_mb"],
                    "selection_evidence": "analytical estimate",
                }
                for row in performance["platform_estimates"]
                if row["platform"] == platform
            ]

        for row in source_rows:
            model = row["model"]
            accuracy = float(row["accuracy"])
            latency = float(row["latency_ms"])
            model_size = float(row["model_size_mb"])
            energy = float(
                row.get("energy_mj", row.get("estimated_energy_mj", 0.0))
            )
            working_set = float(
                row.get(
                    "working_set_mb",
                    row.get("estimated_memory_mb", model_size),
                )
            )
            checks = {
                "accuracy": (
                    None
                    if normalized["accuracy_target"] is None
                    else accuracy >= normalized["accuracy_target"]
                ),
                "latency": (
                    None
                    if normalized["max_latency_ms"] is None
                    else latency <= normalized["max_latency_ms"]
                ),
                "model_size": (
                    None
                    if normalized["max_model_size_mb"] is None
                    else model_size <= normalized["max_model_size_mb"]
                ),
                "energy": (
                    None
                    if normalized["max_energy_mj"] is None
                    else energy <= normalized["max_energy_mj"]
                ),
                "working_set": (
                    None
                    if normalized["max_working_set_mb"] is None
                    else working_set <= normalized["max_working_set_mb"]
                ),
            }
            feasible = all(
                passed for passed in checks.values() if passed is not None
            ) and normalized["is_complete"]
            candidates.append(
                {
                    "model": model,
                    "accuracy": accuracy,
                    "latency_ms": latency,
                    "model_size_mb": model_size,
                    "estimated_energy_mj": energy,
                    "working_set_mb": working_set,
                    "checks": checks,
                    "feasible": feasible,
                }
            )

        feasible_candidates = [
            candidate for candidate in candidates if candidate["feasible"]
        ]
        feasible_candidates.sort(
            key=lambda candidate: (
                -candidate["accuracy"],
                candidate["latency_ms"],
                candidate["model_size_mb"],
            )
        )
        return {
            "base_model": (
                base_model.get("name", "provided model")
                if isinstance(base_model, dict)
                else str(base_model)
            ),
            "constraints": normalized,
            "candidates": candidates,
            "feasible_models": [
                candidate["model"] for candidate in feasible_candidates
            ],
            "recommended_model": (
                feasible_candidates[0]["model"]
                if feasible_candidates
                else None
            ),
            "decision_rule": (
                "filter by hard constraints, then prefer accuracy and "
                "lower latency among feasible variants"
            ),
        }


def build_reference_design_cases() -> list[dict]:
    """Build mobile and constrained-MCU design examples."""
    methodology = HardwareAwareDesignMethodology()
    cases = [
        {
            "name": "accuracy-sensitive mobile deployment",
            "requirements": {
                "platform": "arm_cortex_a78",
                "accuracy_target": 0.88,
                "max_latency_ms": 20.0,
                "max_model_size_mb": 10.0,
                "max_energy_mj": 20.0,
                "max_working_set_mb": 8.0,
            },
        },
        {
            "name": "strict SRAM MCU deployment",
            "requirements": {
                "platform": "arm_cortex_m7",
                "accuracy_target": 0.85,
                "max_latency_ms": 1000.0,
                "max_model_size_mb": 1.0,
                "max_energy_mj": 40.0,
                "max_working_set_mb": 0.5,
            },
        },
    ]
    results = []
    for case in cases:
        constraints = methodology.constraint_specification_framework(
            case["requirements"]
        )
        results.append(
            {
                "name": case["name"],
                "constraints": constraints,
                "priorities": methodology.optimization_priority_framework(
                    constraints
                ),
                "design_space": methodology.design_space_exploration(
                    "MobileNetV2 baseline",
                    constraints,
                ),
            }
        )
    return results
