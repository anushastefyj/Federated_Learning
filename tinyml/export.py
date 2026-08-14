import torch
import torch.nn as nn
import os
import numpy as np

def export_pytorch_to_onnx(model: nn.Module, seq_length: int, num_features: int, output_path: str):
    """
    Exports a PyTorch model to ONNX format.
    This is the first step to converting to TFLite for TinyML.
    (PyTorch -> ONNX -> TensorFlow -> TFLite)
    """
    model.eval()
    
    # Create dummy input matching the expected shape: (batch, seq_length, features)
    dummy_input = torch.randn(1, seq_length, num_features)
    
    print(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export complete.")


def get_keras_model(model_type: str, seq_length: int, num_features: int, hidden_dim: int):
    """
    Since TFLite Micro natively supports Keras/TensorFlow best, providing a Keras equivalent
    of the PyTorch models ensures a robust path for deployment.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    if model_type.lower() == "mlp":
        model = models.Sequential([
            layers.InputLayer(input_shape=(seq_length, num_features)),
            layers.Flatten(),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(1)
        ])
        return model
    elif model_type.lower() == "cnn":
        model = models.Sequential([
            layers.InputLayer(input_shape=(seq_length, num_features)),
            # Keras Conv1D takes (batch, seq_length, channels), PyTorch takes (batch, channels, seq_length)
            # The data here comes as (seq, feat) so it works naturally with Keras.
            layers.Conv1D(filters=hidden_dim, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(1)
        ])
        return model
    raise ValueError(f"Unknown type: {model_type}")

def convert_keras_to_tflite_int8(keras_model, output_path: str, representative_data_gen=None):
    """
    Converts a Keras model to TFLite with INT8 Post-Training Quantization (PTQ).
    """
    import tensorflow as tf
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # Enable optimizations for size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # To ensure fully INT8 quantization (required for some microcontrollers),
    # we need a representative dataset generator to calibrate activations.
    if representative_data_gen is not None:
        converter.representative_dataset = representative_data_gen
        # Restrict supported ops to INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    size_kb = len(tflite_model) / 1024
    print(f"TFLite INT8 model saved to {output_path} (Size: {size_kb:.2f} KB)")
    
    return tflite_model

def get_representative_dataset_generator(dataloader, num_samples=100):
    """
    Creates a generator for TFLite quantization calibration using the PyTorch DataLoader.
    """
    import tensorflow as tf
    def representative_data_gen():
        count = 0
        for X, _ in dataloader:
            # X is PyTorch tensor, convert to numpy float32
            x_np = X.numpy().astype(np.float32)
            for i in range(x_np.shape[0]):
                yield [np.expand_dims(x_np[i], axis=0)]
                count += 1
                if count >= num_samples:
                    return
    return representative_data_gen
