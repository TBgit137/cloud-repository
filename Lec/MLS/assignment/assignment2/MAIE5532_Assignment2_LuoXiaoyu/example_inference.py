"""Run the final cloud, edge, or microcontroller model on one CIFAR-10 image."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "part4/part4_results/optimized_models"
MODEL_FILES = {
    "cloud_server": "cloud_server.keras",
    "edge_device": "edge_device.tflite",
    "microcontroller": "microcontroller.tflite",
}
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def configure_runtime():
    """Avoid reserving all GPU memory when loading the cloud model."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_test_data():
    """Use the same normalization as training; download CIFAR-10 if needed."""
    (_, _), (images, labels) = tf.keras.datasets.cifar10.load_data()
    return images.astype(np.float32) / 255.0, labels.reshape(-1)


class DeploymentPredictor:
    """Share model loading and single-image inference with the benchmark."""

    def __init__(self, target):
        self.path = MODEL_DIR / MODEL_FILES[target]
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing model: {self.path}. Run Part 4 first.")
        self.target = target
        if target == "cloud_server":
            self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            self.runtime = f"Local TensorFlow Keras on {self.device}"
            with tf.device(self.device):
                self.model = tf.keras.models.load_model(self.path, compile=False)
        else:
            self.runtime = "Local TensorFlow Lite CPU, one thread; not target-board hardware"
            self.interpreter = tf.lite.Interpreter(model_path=str(self.path), num_threads=1)
            self.interpreter.allocate_tensors()
            self.input_info = self.interpreter.get_input_details()[0]
            self.output_info = self.interpreter.get_output_details()[0]

    def predict(self, image):
        """Return class scores for one normalized 32x32 RGB image."""
        value = np.asarray(image, dtype=np.float32)[None, ...]
        if self.target == "cloud_server":
            with tf.device(self.device):
                # Converting to NumPy also waits for GPU output to reach the host.
                return self.model(value, training=False).numpy()[0]

        if np.issubdtype(self.input_info["dtype"], np.integer):
            scale, zero = self.input_info["quantization"]
            limits = np.iinfo(self.input_info["dtype"])
            value = np.clip(np.round(value / scale + zero), limits.min, limits.max)
        self.interpreter.set_tensor(
            self.input_info["index"], value.astype(self.input_info["dtype"]))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_info["index"])
        if np.issubdtype(self.output_info["dtype"], np.integer):
            scale, zero = self.output_info["quantization"]
            output = (output.astype(np.float32) - zero) * scale
        return output[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["all", *MODEL_FILES], default="all")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.sample_index < 10000:
        parser.error("--sample-index must be between 0 and 9999.")
    configure_runtime()
    images, labels = load_test_data()
    image, label = images[args.sample_index], int(labels[args.sample_index])
    print(f"Test image: {args.sample_index}; true label: {CLASS_NAMES[label]}")
    targets = MODEL_FILES if args.target == "all" else [args.target]
    for target in targets:
        predictor = DeploymentPredictor(target)
        scores = predictor.predict(image)
        prediction = int(np.argmax(scores))
        print(f"{target}: prediction={CLASS_NAMES[prediction]}, "
              f"score={scores[prediction]:.4f}, correct={prediction == label}")
        print(f"  Runtime: {predictor.runtime}")


if __name__ == "__main__":
    main()
