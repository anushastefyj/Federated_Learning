import flwr as fl
from typing import Dict, List, Tuple, Callable
import torch
from .client import AirQualityClient
from .strategies import CalibrationAwareStrategy

def get_on_fit_config_fn(local_epochs: int, mu: float) -> Callable:
    """Return a function which returns training configurations."""
    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "mu": mu,
        }
    return fit_config

def start_fl_simulation(
    num_clients: int,
    client_loaders: List[Tuple],
    model_fn: Callable,
    strategy_name: str = "fedavg",
    num_rounds: int = 10,
    fraction_fit: float = 1.0,
    local_epochs: int = 3,
    mu: float = 0.01,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu")
):
    """
    Starts the Flower simulation.
    """
    
    # Factory to create clients for simulation
    def client_fn(cid: str) -> fl.client.Client:
        cid = int(cid)
        train_loader, val_loader = client_loaders[cid]
        model = model_fn()
        return AirQualityClient(model, train_loader, val_loader, device, lr).to_client()

    # Determine strategy
    on_fit_config = get_on_fit_config_fn(local_epochs, mu)
    
    if strategy_name.lower() == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            on_fit_config_fn=on_fit_config,
        )
    elif strategy_name.lower() == "fedprox":
        # In Flower, FedProx is implemented by applying proximal term locally (which we do in client.py)
        # Server-side is the same as FedAvg. But Flower has a built-in FedProx we can use.
        strategy = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            proximal_mu=mu,
            on_fit_config_fn=on_fit_config,
        )
    elif strategy_name.lower() == "calibration_aware":
        strategy = CalibrationAwareStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            on_fit_config_fn=on_fit_config,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy_name}")

    print(f"--- Starting FL Simulation with {strategy_name.upper()} ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return history
