"""Benchmark the three final deployment models without retraining."""

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from example_inference import (
    ROOT, MODEL_FILES, DeploymentPredictor, configure_runtime, load_test_data,
)


WARMUP_RUNS = 20
MEASURED_RUNS = 100


def benchmark(target, images, labels):
    predictor = DeploymentPredictor(target)
    print(f"Benchmarking {target}: {len(labels)} test images", flush=True)
    correct = 0
    for index, (image, label) in enumerate(zip(images, labels), start=1):
        correct += int(np.argmax(predictor.predict(image)) == int(label))
        if index % 2000 == 0:
            print(f"  Evaluated {index}/{len(labels)} images", flush=True)

    # Time the same batch-size-1 call used by the inference example.
    for index in range(WARMUP_RUNS):
        predictor.predict(images[index % len(images)])
    durations = []
    for index in range(MEASURED_RUNS):
        image = images[index % len(images)]
        start = time.perf_counter()
        predictor.predict(image)
        durations.append((time.perf_counter() - start) * 1000)

    mean_ms = float(np.mean(durations))
    return {
        "model_path": str(predictor.path.relative_to(ROOT)),
        "runtime": predictor.runtime,
        "test_accuracy": correct / len(labels),
        "test_samples": len(labels),
        "model_size_bytes": predictor.path.stat().st_size,
        "model_size_mib": predictor.path.stat().st_size / (1024 ** 2),
        "batch_size": 1,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": float(np.median(durations)),
        "latency_p95_ms": float(np.percentile(durations, 95)),
        "serial_inference_images_per_second": 1000 / mean_ms,
        "inference_memory_mib": None,
        "power_mw": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["all", *MODEL_FILES], default="all")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Test images to evaluate; use 100 for a quick check.")
    args = parser.parse_args()
    if not 1 <= args.samples <= 10000:
        parser.error("--samples must be between 1 and 10000.")
    configure_runtime()
    images, labels = load_test_data()
    targets = MODEL_FILES if args.target == "all" else [args.target]
    results = {}
    for target in targets:
        results[target] = benchmark(target, images[:args.samples], labels[:args.samples])
        result = results[target]
        print(f"{target}: accuracy={result['test_accuracy']:.2%}, "
              f"size={result['model_size_mib']:.4f} MiB, "
              f"mean latency={result['latency_mean_ms']:.3f} ms")

    report = {
        "tensorflow_version": tf.__version__,
        "results": results,
        "timing_scope": "Single-image prediction including input/output handling and host output transfer; excludes model loading, dataset loading, and normalization.",
        "throughput_scope": "Reciprocal of mean serial batch-size-1 latency, not batched or concurrent serving throughput.",
        "hardware_scope": "Local execution only. Cloud uses GPU when available; edge and microcontroller models use the local CPU. Latencies are not target-board measurements or cross-device speedup ratios.",
        "memory_power_scope": "Inference memory and power were not measured; null does not mean zero.",
    }
    output = ROOT / "benchmark_results"
    output.mkdir(exist_ok=True)
    # Keep quick checks separate from full test-set benchmark results.
    path = output / f"deployment_benchmark_{args.target}_{args.samples}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark saved to: {path}")


if __name__ == "__main__":
    main()
