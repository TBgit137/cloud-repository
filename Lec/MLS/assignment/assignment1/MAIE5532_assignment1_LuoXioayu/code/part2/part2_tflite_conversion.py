import os
import numpy as np
import tensorflow as tf

def load_and_preprocess_data():
    """
    Load and preprocess MNIST test data for inference validation and representative dataset calibration.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    return x_train, y_train, x_test, y_test

def convert_to_tflite(model_path, quantize=False, x_train=None):
    """
    Convert TensorFlow/Keras model to TensorFlow Lite format.
    
    Args:
        model_path (str): Path to saved .keras model
        quantize (bool): Enable full integer (int8) quantization
        x_train (ndarray): Representative dataset for quantization calibration
        
    Returns:
        bytes: Converted TFLite binary data
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_data_gen():
            if x_train is None:
                raise ValueError("Full integer quantization requires x_train for calibration!")
            for i in range(100):
                sample = np.expand_dims(x_train[i], axis=0)
                yield [sample]
                
        converter.representative_dataset = representative_data_gen
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
    # Convert and return model binary data
    tflite_model_data = converter.convert()
    return tflite_model_data

def analyze_model_size(tf_model_path, tflite_model_data, model_name="TFLite Model"):
    """
    Compare Keras and TFLite model file sizes and compression ratio.
    """
    keras_size = os.path.getsize(tf_model_path)
    tflite_size = len(tflite_model_data)
    
    compression_ratio = keras_size / tflite_size
    size_reduction = (1 - (tflite_size / keras_size)) * 100
    
    print(f"\n================ [{model_name} Size Analysis] ================")
    print(f"Original Keras model size:    {keras_size / 1024:.2f} KB ({keras_size} bytes)")
    print(f"Converted TFLite model size:  {tflite_size / 1024:.2f} KB ({tflite_size} bytes)")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Size Reduction:    {size_reduction:.2f}%")

def test_tflite_accuracy(tflite_model_data, x_test, y_test):
    """
    Evaluate model accuracy on test set using TFLite Interpreter.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()
    
    # Get tensor metadata
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_index = input_details['index']
    output_index = output_details['index']
    input_type = input_details['dtype']
    
    correct_count = 0
    total_samples = len(x_test)
    
    # Run inference sample by sample
    for i in range(total_samples):
        input_data = np.expand_dims(x_test[i], axis=0)
        
        # Manual quantization for int8 models
        if input_type == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = np.round(input_data / scale) + zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_index)
        predicted_label = np.argmax(output_data[0])
        if predicted_label == y_test[i]:
            correct_count += 1
            
    accuracy = correct_count / total_samples
    
    if input_type == np.int8:
        print("\n---------------- [Quantization Tensor Metadata] ----------------")
        print(f"Input Tensor -> Scale: {input_details['quantization'][0]:.6f}, Zero Point: {input_details['quantization'][1]}")
        print(f"Output Tensor -> Scale: {output_details['quantization'][0]:.6f}, Zero Point: {output_details['quantization'][1]}")
        
    return accuracy

if __name__ == "__main__":
    model_path = os.path.join("..", "part1", "mnist_cnn_model.keras")
    
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    print(f"Loading model file: {model_path}...")
    
    # 1. Convert to Float32 TFLite model
    print("Converting to Float32 TFLite model...")
    tflite_model = convert_to_tflite(model_path, quantize=False)
    
    # 2. Convert to Int8 quantized TFLite model
    print("Converting to Int8 quantized TFLite model...")
    tflite_quantized_model = convert_to_tflite(model_path, quantize=True, x_train=x_train)
    
    # 3. Save TFLite models
    with open('mnist_model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    with open('mnist_model_quantized.tflite', 'wb') as f:
        f.write(tflite_quantized_model)
        
    print("\nModel files saved successfully: 'mnist_model.tflite' and 'mnist_model_quantized.tflite'")
    
    # 4. Compare model sizes
    analyze_model_size(model_path, tflite_model, model_name="Float32 TFLite")
    analyze_model_size(model_path, tflite_quantized_model, model_name="Int8 Quantized TFLite")
    
    # 5. Test converted model accuracy
    print("\nEvaluating Float32 TFLite model accuracy...")
    tflite_accuracy = test_tflite_accuracy(tflite_model, x_test, y_test)
    
    print("\nEvaluating Int8 Quantized TFLite model accuracy...")
    tflite_quantized_accuracy = test_tflite_accuracy(tflite_quantized_model, x_test, y_test)
    
    print("\n================ [Final Accuracy Comparison] ================")
    print(f"TensorFlow Lite (Float32) accuracy:   {tflite_accuracy:.4f}")
    print(f"TensorFlow Lite (Int8 quantized) accuracy: {tflite_quantized_accuracy:.4f}")