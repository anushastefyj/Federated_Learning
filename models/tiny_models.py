import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMLP(nn.Module):
    """
    Ultra-low parameter Multi-Layer Perceptron.
    Suitable for microcontrollers.
    """
    def __init__(self, seq_length: int, num_features: int, hidden_dim: int = 16):
        super(TinyMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(seq_length * num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Squeeze the output to match target shape (batch_size,)
        return x.squeeze(-1)

class TinyCNN(nn.Module):
    """
    Ultra-low parameter 1D Convolutional Neural Network.
    Captures temporal dependencies efficiently.
    """
    def __init__(self, seq_length: int, num_features: int, hidden_dim: int = 16):
        super(TinyCNN, self).__init__()
        # PyTorch Conv1d expects (batch_size, channels, seq_length)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened dimension
        self.flattened_dim = hidden_dim * (seq_length // 2)
        
        self.fc1 = nn.Linear(self.flattened_dim, 1)

    def forward(self, x):
        # Input x shape: (batch_size, seq_length, num_features)
        # Permute to: (batch_size, num_features, seq_length) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        
        return x.squeeze(-1)

def get_model(model_type: str, seq_length: int, num_features: int, hidden_dim: int) -> nn.Module:
    """Factory function to get the requested model."""
    if model_type.lower() == "mlp":
        return TinyMLP(seq_length, num_features, hidden_dim)
    elif model_type.lower() == "cnn":
        return TinyCNN(seq_length, num_features, hidden_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
