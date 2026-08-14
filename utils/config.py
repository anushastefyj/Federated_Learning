import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # General
    seed: int = 42
    output_dir: str = "./outputs"
    
    # Data
    data_path: str = "./data/raw/sensor_data.csv"
    seq_length: int = 24  # e.g., 24 hours of historical data
    pred_horizon: int = 1 # predict next hour
    batch_size: int = 32
    num_clients: int = 5
    
    # Model
    model_type: str = "cnn" # "cnn" or "mlp"
    hidden_dim: int = 16
    
    # Federated Learning
    num_rounds: int = 10
    fraction_fit: float = 1.0 # Sample all clients
    local_epochs: int = 3
    learning_rate: float = 0.001
    strategy: str = "fedavg" # "fedavg", "fedprox", "calibration_aware"
    mu: float = 0.01 # Proximal term for FedProx/Calibration-Aware
    
    # TinyML Export
    quantization: str = "int8"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

# Global configuration instance
config = Config()
