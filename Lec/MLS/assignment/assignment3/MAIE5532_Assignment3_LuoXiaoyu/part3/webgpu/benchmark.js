(async () => {
  const statusElement = document.getElementById("status");
  const resultElement = document.getElementById("result");

  try {
    if (!navigator.gpu) {
      throw new Error("navigator.gpu is unavailable in this browser");
    }

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

    const session = await ort.InferenceSession.create(
      "../models/latency_optimized.onnx",
      {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      },
    );

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const inputShape = [1, 160, 160, 3];
    const inputSize = inputShape.reduce((product, value) => product * value, 1);
    const source = new Float32Array(inputSize);
    for (let index = 0; index < source.length; index += 1) {
      source[index] = (index % 256);
    }
    const input = new ort.Tensor("float32", source, inputShape);

    let output = null;
    for (let index = 0; index < 10; index += 1) {
      output = await session.run({ [inputName]: input });
    }

    const timings = [];
    for (let index = 0; index < 100; index += 1) {
      const start = performance.now();
      output = await session.run({ [inputName]: input });
      timings.push(performance.now() - start);
    }

    timings.sort((left, right) => left - right);
    const mean = timings.reduce((sum, value) => sum + value, 0) / timings.length;
    const result = {
      backend: "WebGPU proxy",
      runtime: "ONNX Runtime Web 1.23.2",
      input_name: inputName,
      output_name: outputName,
      output_shape: output[outputName].dims,
      warmup_runs: 10,
      measured_runs: 100,
      mean_ms: mean,
      median_ms: timings[Math.floor(timings.length / 2)],
      p95_ms: timings[Math.floor(timings.length * 0.95)],
    };
    statusElement.textContent = "Benchmark complete";
    resultElement.textContent = JSON.stringify(result, null, 2);
    document.title = `WebGPU benchmark complete: ${mean.toFixed(3)} ms`;
    window.webgpuBenchmarkResult = result;
  } catch (error) {
    statusElement.textContent = "Benchmark failed";
    resultElement.textContent = JSON.stringify(
      { error: String(error), stack: error.stack },
      null,
      2,
    );
    document.title = `WebGPU benchmark failed: ${error}`;
  }
})();
