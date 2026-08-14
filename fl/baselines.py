import torch
import torch.nn as nn
from .client import train, test
from typing import List, Tuple

def run_centralized_baseline(
    model: nn.Module,
    client_loaders: List[Tuple],
    epochs: int,
    lr: float,
    device: torch.device
):
    """
    Simulates a centralized baseline where all client data is conceptually pooled.
    In a real scenario, you'd concatenate the datasets. Here we iterate sequentially
    over all loaders in each epoch to simulate pooled training.
    """
    print("--- Starting Centralized Baseline ---")
    
    # We will just sequentially train on each client's train loader per epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for cid, (train_loader, _) in enumerate(client_loaders):
            train(model, train_loader, epochs=1, lr=lr, device=device)
            
    # Evaluate across all validation sets
    total_loss = 0.0
    for cid, (_, val_loader) in enumerate(client_loaders):
        loss = test(model, val_loader, device=device)
        total_loss += loss
        
    avg_loss = total_loss / len(client_loaders)
    print(f"Centralized Baseline - Average Validation Loss: {avg_loss:.4f}")
    return avg_loss

def run_local_baseline(
    model_fn,
    client_loaders: List[Tuple],
    epochs: int,
    lr: float,
    device: torch.device
):
    """
    Simulates a local-only baseline where each client trains a model independently.
    """
    print("--- Starting Local-Only Baseline ---")
    local_losses = []
    
    for cid, (train_loader, val_loader) in enumerate(client_loaders):
        model = model_fn().to(device)
        # Train locally
        train(model, train_loader, epochs=epochs, lr=lr, device=device)
        # Evaluate locally
        loss = test(model, val_loader, device=device)
        print(f"Client {cid} - Local Validation Loss: {loss:.4f}")
        local_losses.append(loss)
        
    avg_loss = sum(local_losses) / len(local_losses)
    print(f"Local-Only Baseline - Average Validation Loss: {avg_loss:.4f}")
    return avg_loss
