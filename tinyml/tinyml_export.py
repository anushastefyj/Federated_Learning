import os
import torch
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import TinyMLP, TinyCNN1D

def export_to_tflite(pytorch_model, input_shape, save_dir, model_name="tinyml_model"):
    """
    Exports a trained PyTorch model through ONNX into TensorFlow SavedModel,
    and then generates both a Float TFLite and an INT8 Quantized TFLite model.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n--- Exporting {model_name} ---")
    
    # 1. PyTorch to ONNX
    onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
    dummy_input = torch.randn(1, *input_shape, dtype=torch.float32)
    
    torch.onnx.export(
        pytorch_model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None # Fix dimensions for TinyML
    )
    print(f"[1/4] Exported ONNX to {onnx_path}")
    
    # 2. ONNX to TF SavedModel
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("ERROR: Required libraries missing for TFLite conversion.")
        print("Please run: pip install onnx onnx-tf tensorflow")
        return
        
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    
    saved_model_dir = os.path.join(save_dir, f"{model_name}_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"[2/4] Exported TF SavedModel to {saved_model_dir}")
    
    # 3. Float TFLite Model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_float_model = converter.convert()
    float_tflite_path = os.path.join(save_dir, f"{model_name}_float.tflite")
    with open(float_tflite_path, "wb") as f:
        f.write(tflite_float_model)
    print(f"[3/4] Exported Float TFLite to {float_tflite_path}")
    
    # 4. INT8 Quantized TFLite Model
    converter_quant = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for calibrating INT8 ranges
    def representative_dataset():
        for _ in range(100):
            yield [np.random.randn(1, *input_shape).astype(np.float32)]
            
    converter_quant.representative_dataset = representative_dataset
    
    # Restrict to strictly INT8 ops (required for microcontrollers / Coral Edge TPUs)
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.inference_input_type = tf.int8
    converter_quant.inference_output_type = tf.int8
    
    try:
        tflite_quant_model = converter_quant.convert()
        quant_tflite_path = os.path.join(save_dir, f"{model_name}_quant_int8.tflite")
        with open(quant_tflite_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"[4/4] Exported Quantized INT8 TFLite to {quant_tflite_path}")
    except Exception as e:
        print(f"[4/4] INT8 Quantization failed: {e}")


if __name__ == "__main__":
    print("="*60)
    print(" TINYML EXPORT INSTRUCTIONS FOR GOOGLE COLAB")
    print("="*60)
    print("Run this cell first to ensure dependencies exist:")
    print("!pip install torch torchvision torchaudio onnx onnx-tf tensorflow")
    print("\nRunning example export...")
    
    # Generate dummy untraind models to demonstrate the pipeline
    WINDOW_SIZE = 12
    NUM_FEATURES = 8
    
    mlp = TinyMLP(window_size=WINDOW_SIZE, num_features=NUM_FEATURES, return_aqi=False)
    cnn = TinyCNN1D(window_size=WINDOW_SIZE, num_features=NUM_FEATURES, return_aqi=False)
    
    mlp.eval()
    cnn.eval()
    
    export_to_tflite(mlp, (WINDOW_SIZE, NUM_FEATURES), save_dir="export_models", model_name="tinymlp")
    export_to_tflite(cnn, (WINDOW_SIZE, NUM_FEATURES), save_dir="export_models", model_name="tinycnn1d")
