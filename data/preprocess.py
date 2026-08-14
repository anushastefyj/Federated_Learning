import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_samples: int = 10000, num_features: int = 3) -> pd.DataFrame:
    """
    Generates synthetic time-series air quality data if real data is absent.
    Features: PM2.5, Temperature, Humidity
    """
    np.random.seed(42)
    time = np.arange(num_samples)
    
    # Simulate a daily cycle + noise
    daily_cycle = np.sin(2 * np.pi * time / 24)
    
    pm25 = 50 + 20 * daily_cycle + np.random.normal(0, 10, num_samples)
    temp = 25 + 5 * daily_cycle + np.random.normal(0, 2, num_samples)
    humidity = 60 - 10 * daily_cycle + np.random.normal(0, 5, num_samples)
    
    # Ensure no negative PM2.5 or humidity
    pm25 = np.clip(pm25, 0, None)
    humidity = np.clip(humidity, 0, 100)
    
    df = pd.DataFrame({
        "PM2.5": pm25,
        "Temperature": temp,
        "Humidity": humidity
    })
    return df

def load_and_preprocess(data_path: str) -> np.ndarray:
    """
    Loads raw CSV data, imputes missing values, and normalizes.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Assumes target column is PM2.5 and numeric features are present
        # Drop timestamps or non-numeric if necessary (customize later)
        df = df.select_dtypes(include=[np.number])
    else:
        print(f"Warning: {data_path} not found. Generating synthetic data.")
        df = generate_synthetic_data()

    # Impute missing values (forward fill then backward fill)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
    # Normalization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    return scaled_data

def create_non_iid_splits(data: np.ndarray, num_clients: int) -> List[np.ndarray]:
    """
    Partitions the dataset into `num_clients` subsets.
    To simulate Non-IID sensor data, we split the data and apply artificial scaling
    to simulate different 'pollution regimes' (e.g., industrial vs residential zones).
    """
    splits = np.array_split(data, num_clients)
    
    non_iid_splits = []
    for i, split in enumerate(splits):
        # Apply a regime multiplier to the PM2.5 column (assume idx 0 is target)
        # Client 0 gets 0.5x pollution, Client num_clients-1 gets 1.5x pollution
        multiplier = 0.5 + (i / max(1, num_clients - 1))
        
        modified_split = split.copy()
        modified_split[:, 0] *= multiplier
        non_iid_splits.append(modified_split)
        
    return non_iid_splits

def get_client_dataloaders(data_path: str, num_clients: int, seq_length: int, batch_size: int, test_split: float = 0.2):
    """
    Returns train and validation dataloaders for all clients.
    """
    from torch.utils.data import DataLoader
    from .dataset import AirQualityDataset
    
    full_data = load_and_preprocess(data_path)
    client_data_splits = create_non_iid_splits(full_data, num_clients)
    
    client_loaders = []
    
    for split in client_data_splits:
        split_idx = int(len(split) * (1 - test_split))
        train_data = split[:split_idx]
        val_data = split[split_idx:]
        
        train_ds = AirQualityDataset(train_data, seq_length=seq_length)
        val_ds = AirQualityDataset(val_data, seq_length=seq_length)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        client_loaders.append((train_loader, val_loader))
        
    return client_loaders
