#include <cstdint>
#include <ctime>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const unsigned char g_cifar_model[];
extern const unsigned int g_cifar_model_len;

namespace {

using CifarOpResolver = tflite::MicroMutableOpResolver<8>;

TfLiteStatus RegisterOps(CifarOpResolver& resolver) {
  TF_LITE_ENSURE_STATUS(resolver.AddMul());
  TF_LITE_ENSURE_STATUS(resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(resolver.AddPad());
  TF_LITE_ENSURE_STATUS(resolver.AddMean());
  TF_LITE_ENSURE_STATUS(resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(resolver.AddSoftmax());
  return kTfLiteOk;
}

}  // namespace

int main(int argc, char** argv) {
  tflite::InitializeTarget();
  (void)argc;
  (void)argv;

  const tflite::Model* model = tflite::GetModel(g_cifar_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Unsupported TFLite schema version.");
    return 1;
  }

  CifarOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    MicroPrintf("Failed to register CIFAR model operators.");
    return 1;
  }

  // Keep the arena at an MCU-scale limit to test whether the model fits in SRAM.
  alignas(16) static uint8_t tensor_arena[1024 * 1024];
  tflite::MicroProfiler profiler;
  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, sizeof(tensor_arena), nullptr, &profiler);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Failed to allocate tensors.");
    return 1;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input == nullptr || output == nullptr) {
    MicroPrintf("Model does not have the expected input/output tensor.");
    return 1;
  }

  for (int index = 0; index < input->bytes; ++index) {
    input->data.int8[index] = static_cast<int8_t>((index % 17) - 8);
  }

  for (int index = 0; index < 2; ++index) {
    if (interpreter.Invoke() != kTfLiteOk) {
      MicroPrintf("Warm-up inference failed.");
      return 1;
    }
  }

  constexpr int kMeasuredRuns = 5;
  const std::clock_t clock_start = std::clock();
  for (int index = 0; index < kMeasuredRuns; ++index) {
    if (interpreter.Invoke() != kTfLiteOk) {
      MicroPrintf("Measured inference failed.");
      return 1;
    }
  }
  const std::clock_t clock_elapsed = std::clock() - clock_start;
  const double average_clock_ms =
      1000.0 * static_cast<double>(clock_elapsed) / CLOCKS_PER_SEC;

  MicroPrintf("CIFAR model bytes: %u", g_cifar_model_len);
  MicroPrintf("Input bytes: %u", input->bytes);
  MicroPrintf("Output bytes: %u", output->bytes);
  MicroPrintf("Tensor arena bytes: %u", sizeof(tensor_arena));
  MicroPrintf("Tensor arena used bytes: %u",
              interpreter.arena_used_bytes());
  MicroPrintf("Measured runs: %d", kMeasuredRuns);
  MicroPrintf("Clock ticks per second: %d", CLOCKS_PER_SEC);
  MicroPrintf("Average clock milliseconds: %.3f",
              average_clock_ms / kMeasuredRuns);
  MicroPrintf("~~~QEMU CIFAR BENCHMARK PASSED~~~");
  return 0;
}
