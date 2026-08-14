import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import sys

# Add root to path so we can import models dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import TinyMLP, TinyCNN1D

class StationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_station_data(data_dir, station_id, window_size, num_features):
    """
    Loads Parquet files for a specific station, applies per-station local normalization 
    using StandardScaler fitted ONLY on the train set (no test leakage), 
    and reshapes data for the PyTorch models.
    """
    station_dir = os.path.join(data_dir, station_id)
    
    train_df = pd.read_parquet(os.path.join(station_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(station_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(station_dir, "test.parquet"))
    
    feature_cols = [c for c in train_df.columns if '_lag_' in c]
    target_col = 'target_pm25'
    
    scaler = StandardScaler()
    
    train_X_flat = scaler.fit_transform(train_df[feature_cols].values)
    val_X_flat = scaler.transform(val_df[feature_cols].values)
    test_X_flat = scaler.transform(test_df[feature_cols].values)
    
    train_y = train_df[target_col].values
    val_y = val_df[target_col].values
    test_y = test_df[target_col].values
    
    train_X = train_X_flat.reshape(-1, window_size, num_features)
    val_X = val_X_flat.reshape(-1, window_size, num_features)
    test_X = test_X_flat.reshape(-1, window_size, num_features)
    
    train_dataset = StationDataset(train_X, train_y)
    val_dataset = StationDataset(val_X, val_y)
    test_dataset = StationDataset(test_X, test_y)
    
    return train_dataset, val_dataset, test_dataset

def train(model, train_loader, epochs, lr, device, global_params=None, proximal_mu=0.0):
    """Local training loop, augmented with FedProx proximal term if requested."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            if isinstance(outputs, tuple): 
                outputs = outputs[0]
            
            loss = criterion(outputs, y)
            
            # --- FedProx Proximal Term ---
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += torch.square(local_weights - global_weights).sum()
                loss += (proximal_mu / 2.0) * proximal_term
                
            loss.backward()
            optimizer.step()

def test(model, test_loader, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, y)
            total_loss += loss.item() * len(X)
            total_samples += len(X)
    
    mse = total_loss / total_samples
    rmse = float(np.sqrt(mse))
    return mse, rmse

def get_calibration_stats(model, loader, device):
    """
    Computes simple calibration statistics for the current model on the given dataset.
    Returns: Mean Error (Bias), Mean Absolute Error (MAE)
    """
    model.eval()
    errors = []
    abs_errors = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # predictions - targets
            err = (outputs - y).cpu().numpy().flatten()
            errors.extend(err)
            abs_errors.extend(np.abs(err))
            
    return float(np.mean(errors)), float(np.mean(abs_errors))

class AirQualityClient(fl.client.NumPyClient):
    """
    Flower NumPyClient that wraps the PyTorch training/evaluation loop for a single station.
    """
    def __init__(self, station_id, data_dir, model_name, window_size, num_features, batch_size=32, lr=1e-3, epochs=1):
        self.station_id = station_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr
        
        if model_name.lower() == 'mlp':
            self.model = TinyMLP(window_size, num_features).to(self.device)
        elif model_name.lower() == 'cnn':
            self.model = TinyCNN1D(window_size, num_features).to(self.device)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
            
        train_ds, val_ds, _ = load_station_data(data_dir, station_id, window_size, num_features)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size)
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Local training augmented with calibration tracking."""
        self.set_parameters(parameters)
        
        # FedProx proximal_mu is typically passed down via the config dictionary
        proximal_mu = config.get("proximal_mu", 0.0)
        
        # Clone global params to anchor the proximal term
        global_params = [p.clone().detach().to(self.device) for p in self.model.parameters()]
        
        train(self.model, self.train_loader, self.epochs, self.lr, self.device, global_params, proximal_mu)
        
        # Compute local calibration stats before returning
        mean_error, mae = get_calibration_stats(self.model, self.val_loader, self.device)
        
        metrics = {
            "val_mean_error": mean_error,
            "val_mae": mae,
        }
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Local evaluation on the client side."""
        self.set_parameters(parameters)
        mse, rmse = test(self.model, self.val_loader, self.device)
        return float(mse), len(self.val_loader.dataset), {"rmse": rmse}
