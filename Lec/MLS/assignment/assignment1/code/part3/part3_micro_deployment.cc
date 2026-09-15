// model_inference.cc
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model data
extern const unsigned char mnist_model_quantized_tflite[];
extern const unsigned int mnist_model_quantized_tflite_len;

// Global variables
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena Size
const int kTensorArenaSize = 8976;
uint8_t tensor_arena[kTensorArenaSize];

void setup_model() {
    // Load model from data
    model = tflite::GetModel(mnist_model_quantized_tflite);

    // Create ops resolver with required operators
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();

    // Set up interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Verify tensor metadata
    MicroPrintf("\n--- Tensor Metadata (Sanity Check) ---");
    MicroPrintf("Input Tensor  - Type: %d, Byte Size: %d, Scale: %f, Zero Point: %d",
                         input->type, input->bytes, static_cast<double>(input->params.scale), input->params.zero_point);
    MicroPrintf("Output Tensor - Type: %d, Byte Size: %d, Scale: %f, Zero Point: %d",
                         output->type, output->bytes, static_cast<double>(output->params.scale), output->params.zero_point);
}

int run_inference(const int8_t* image_data) {
    // Copy input data to model input tensor
    for (int i = 0; i < input->bytes; ++i) {
        input->data.int8[i] = image_data[i];
    }

    // Invoke interpreter
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return -1;
    }

    // Find class with highest score
    int8_t max_score = -128;
    int max_index = -1;
    for (int i = 0; i < 10; ++i) {
        if (output->data.int8[i] > max_score) {
            max_score = output->data.int8[i];
            max_index = i;
        }
    }
    
    return max_index;
}

int main() {
    // Set up model
    setup_model();

    // Print memory usage
    MicroPrintf("\n--- Memory Usage Analysis ---");
    MicroPrintf("Model Size (Bytes): %d", mnist_model_quantized_tflite_len);
    MicroPrintf("Total Arena Size: %d", kTensorArenaSize);
    MicroPrintf("Arena Used: %d bytes", interpreter->arena_used_bytes());

    // Create test input: 28x28 image
    int8_t test_image[784];
    for (int i = 0; i < 784; ++i) {
        test_image[i] = input->params.zero_point;
    }

    // Draw a vertical white line (simulating handwritten digit "1")
    for (int row = 5; row < 23; ++row) {
        test_image[row * 28 + 14] = 127; 
    }

    // Run inference
    MicroPrintf("\n--- Running Inference ---");
    int predicted_class = run_inference(test_image);

    // Print results
    MicroPrintf("Test Input Result -> Predicted Class: %d", predicted_class);

    return 0;
}