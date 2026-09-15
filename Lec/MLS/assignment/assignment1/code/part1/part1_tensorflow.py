import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_model():
    """
    Create a CNN model for MNIST classification
    """
    model = keras.Sequential([
        # Layer 1: Conv2D, 8 filters, 3x3 kernel, ReLU activation
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Layer 2: MaxPool2D, 2x2 pooling size
        keras.layers.MaxPooling2D((2, 2)),
        # Layer 3: Flatten layer
        keras.layers.Flatten(),
        # Layer 4: Dense layer, 16 units, ReLU activation
        keras.layers.Dense(16, activation='relu'),
        # Layer 5: Dense layer, 10 units, output layer
        keras.layers.Dense(10)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_data():
    """
    Load and preprocess MNIST dataset
    """
    # 1. Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 2. Normalize pixel values to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # 3. Reshape CNN input data by adding channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return x_train, y_train, x_test, y_test

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train model and evaluate performance
    """
    # Train for 5 epochs with validation
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test)
    )
    return history

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Create and train model
    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Save trained model
    model.save('mnist_cnn_model.keras')
    print("\n[SUCCESS] Model saved as mnist_cnn_model.keras")
    
    # Evaluate final performance
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")