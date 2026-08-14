import torch
import torch.nn as nn
import torch.onnx
import os

class TinyMLP(nn.Module):
    """
    A simple Multilayer Perceptron (MLP) for time-series forecasting.
    
    Adjusting Size & Capacity:
    - Width: Increase `hidden_dim` (e.g. 32, 64) for more capacity, or decrease (e.g. 8) to save RAM.
    - Depth: Add more `nn.Linear` layers in `self.shared_features` if the task is complex.
    - Note: This model completely flattens the temporal dimension, which is parameter-intensive 
      if window_size is large.
    """
    def __init__(self, window_size, num_features, hidden_dim=16, num_classes=6, return_aqi=False):
        super(TinyMLP, self).__init__()
        self.return_aqi = return_aqi
        self.flatten = nn.Flatten()
        
        input_dim = window_size * num_features
        
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Regression head for next-hour PM2.5
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        # Classification head for AQI bucket (optional)
        if self.return_aqi:
            self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        x = self.flatten(x)
        features = self.shared_features(x)
        
        pm25_pred = self.regression_head(features)
        
        if self.return_aqi:
            aqi_logits = self.classification_head(features)
            return pm25_pred, aqi_logits
            
        return pm25_pred

class TinyCNN1D(nn.Module):
    """
    A lightweight 1D Convolutional Neural Network.
    
    Adjusting Size & Capacity:
    - Width: Increase `num_filters` for wider feature extraction capacity.
    - Depth: Add another Conv1d + MaxPool1d block.
    - Receptive Field: Increase `kernel_size` (e.g., to 5) to capture larger temporal patterns at the cost of parameters.
    """
    def __init__(self, window_size, num_features, num_filters=16, hidden_dim=16, num_classes=6, return_aqi=False):
        super(TinyCNN1D, self).__init__()
        self.return_aqi = return_aqi
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Dummy pass to calculate the flattened feature dimension dynamically based on window_size
        with torch.no_grad():
            # (batch=1, channels=num_features, length=window_size)
            dummy_input = torch.zeros(1, num_features, window_size)
            conv_out = self.conv_block(dummy_input)
            conv_out_dim = conv_out.view(1, -1).shape[1]
        
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        if self.return_aqi:
            self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # PyTorch Conv1d expects (batch_size, channels, length)
        # Input x is (batch_size, window_size, num_features)
        x = x.transpose(1, 2)  
        
        x = self.conv_block(x)
        features = self.fc_shared(x)
        
        pm25_pred = self.regression_head(features)
        
        if self.return_aqi:
            aqi_logits = self.classification_head(features)
            return pm25_pred, aqi_logits
            
        return pm25_pred

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_model_size_kb(model):
    """
    Estimates the model size in KB assuming FP32 representation (4 bytes per parameter).
    Overhead of graph/architecture not included.
    """
    num_params = count_parameters(model)
    size_bytes = num_params * 4
    return size_bytes / 1024.0

def export_model_to_onnx(model, input_shape, file_path="model.onnx"):
    """
    Exports a PyTorch model to ONNX format.
    Useful for later conversion to TensorFlow Lite (via ONNX-TF) or direct ONNX Runtime deployment.
    """
    model.eval()
    # Create dummy input based on the expected input shape, batch size of 1
    dummy_input = torch.randn(1, *input_shape)
    
    # Determine outputs based on the return_aqi flag
    outputs = ['pm25_pred', 'aqi_logits'] if getattr(model, 'return_aqi', False) else ['pm25_pred']
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=outputs,
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        print(f"Successfully exported ONNX model to: {file_path}")
    except Exception as e:
        print(f"Failed to export to ONNX: {e}")

if __name__ == "__main__":
    # Example hyperparameters based on data_loader default assumptions
    WINDOW_SIZE = 12
    NUM_FEATURES = 8  # e.g., PM2.5, PM10, NO2, SO2, CO, O3, temp, humidity
    
    print("--- TinyMLP Evaluation ---")
    mlp = TinyMLP(window_size=WINDOW_SIZE, num_features=NUM_FEATURES, return_aqi=True)
    mlp_params = count_parameters(mlp)
    mlp_size_kb = estimate_model_size_kb(mlp)
    print(f"TinyMLP Parameters: {mlp_params}")
    print(f"TinyMLP Size Estimate: {mlp_size_kb:.2f} KB")
    
    print("\n--- TinyCNN1D Evaluation ---")
    cnn = TinyCNN1D(window_size=WINDOW_SIZE, num_features=NUM_FEATURES, return_aqi=True)
    cnn_params = count_parameters(cnn)
    cnn_size_kb = estimate_model_size_kb(cnn)
    print(f"TinyCNN1D Parameters: {cnn_params}")
    print(f"TinyCNN1D Size Estimate: {cnn_size_kb:.2f} KB")
    
    print("\n--- Sanity Checks ---")
    # Batch of 4 examples
    dummy_data = torch.randn(4, WINDOW_SIZE, NUM_FEATURES)
    
    mlp_pm25, mlp_aqi = mlp(dummy_data)
    print(f"MLP Output Shape - Regression: {mlp_pm25.shape}, Classification: {mlp_aqi.shape}")
    
    cnn_pm25, cnn_aqi = cnn(dummy_data)
    print(f"CNN Output Shape - Regression: {cnn_pm25.shape}, Classification: {cnn_aqi.shape}")
    
    print("\n--- Export Test ---")
    export_model_to_onnx(cnn, input_shape=(WINDOW_SIZE, NUM_FEATURES), file_path="tiny_cnn1d.onnx")
