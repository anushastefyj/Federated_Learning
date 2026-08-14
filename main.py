import argparse
import torch
import os
from utils.config import config
from utils.logger import CSVLogger
from data.preprocess import get_client_dataloaders
from models.tiny_models import get_model
from fl.server import start_fl_simulation
from fl.baselines import run_centralized_baseline, run_local_baseline

def main():
    parser = argparse.ArgumentParser(description="Calibration-Aware Federated Learning for Air-Quality")
    parser.add_argument("--mode", type=str, default="fedavg", choices=["centralized", "local", "fedavg", "fedprox", "calibration_aware"], help="Execution mode")
    parser.add_argument("--clients", type=int, default=config.num_clients, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=config.num_rounds, help="Number of FL rounds")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger = CSVLogger(config.output_dir)

    # 1. Data Pipeline
    print("Preparing Data Loaders...")
    client_loaders = get_client_dataloaders(
        data_path=config.data_path,
        num_clients=args.clients,
        seq_length=config.seq_length,
        batch_size=config.batch_size
    )
    
    # 2. Model Factory Function
    num_features = 3 # Derived from our synthetic data (PM2.5, Temp, Humidity)
    def model_fn():
        return get_model(config.model_type, config.seq_length, num_features, config.hidden_dim).to(device)

    # 3. Execution based on mode
    results = {"mode": args.mode, "clients": args.clients, "rounds": args.rounds}
    
    if args.mode == "centralized":
        model = model_fn()
        avg_loss = run_centralized_baseline(model, client_loaders, config.num_rounds, config.learning_rate, device)
        results["final_val_loss"] = avg_loss
        
    elif args.mode == "local":
        avg_loss = run_local_baseline(model_fn, client_loaders, config.num_rounds, config.learning_rate, device)
        results["final_val_loss"] = avg_loss
        
    else:
        # Federated Modes (FedAvg, FedProx, Calibration_Aware)
        history = start_fl_simulation(
            num_clients=args.clients,
            client_loaders=client_loaders,
            model_fn=model_fn,
            strategy_name=args.mode,
            num_rounds=args.rounds,
            local_epochs=config.local_epochs,
            mu=config.mu,
            lr=config.learning_rate,
            device=device
        )
        
        # Example of getting the last round's loss (if available)
        # Note: Flower history metrics depends on evaluation config
        results["final_val_loss"] = "check_flwr_logs" 

    # 4. Log results
    logger.log(results)
    print(f"Results logged to {logger.filepath}")
    
    # 5. Export to ONNX (Demonstration)
    print("Demonstrating Model Export (PyTorch -> ONNX)...")
    from tinyml.export import export_pytorch_to_onnx
    sample_model = model_fn()
    export_path = os.path.join(config.output_dir, f"{args.mode}_model.onnx")
    export_pytorch_to_onnx(sample_model, config.seq_length, num_features, export_path)

if __name__ == "__main__":
    main()
