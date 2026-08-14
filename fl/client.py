from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import flwr as fl
import numpy as np

# Standard PyTorch Training Loop (can be used for Centralized or FL)
def train(model: nn.Module, train_loader, epochs: int, lr: float, device: torch.device, mu: float = 0.0, global_model: nn.Module = None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Pre-compute global model parameters for FedProx if mu > 0
    global_params = [p.detach().clone() for p in global_model.parameters()] if (mu > 0 and global_model) else None

    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Add FedProx proximal term
            if mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += torch.square((local_weights - global_weights).norm(2))
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

def test(model: nn.Module, test_loader, device: torch.device):
    criterion = nn.MSELoss()
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss += criterion(outputs, y).item()
    return loss / len(test_loader)

# Flower Client
class AirQualityClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, train_loader, val_loader, device: torch.device, lr: float = 0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Get hyperparameters from config sent by server
        epochs = config.get("local_epochs", 1)
        mu = config.get("mu", 0.0)
        
        # For FedProx, we need a clone of the global model
        global_model = None
        if mu > 0:
            import copy
            global_model = copy.deepcopy(self.model)

        train(self.model, self.train_loader, epochs, self.lr, self.device, mu, global_model)
        
        # In a real calibration-aware scenario, we might calculate a calibration error here.
        # For simulation, we return a mock 'calibration_score' in metrics.
        metrics = {"calibration_score": np.random.uniform(0.5, 1.0)} 
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = test(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"val_loss": float(loss)}
