import os
import json
import argparse
import pandas as pd
import flwr as fl

from fl.fl_client import AirQualityClient
from fl.fl_server import get_strategy

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation for Air Quality")
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Directory with processed station data")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"], help="Model architecture")
    parser.add_argument("--strategy", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="FL strategy")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients selected for fit")
    parser.add_argument("--proximal_mu", type=float, default=0.1, help="Proximal term for FedProx")
    parser.add_argument("--window_size", type=int, default=12, help="Time window size")
    parser.add_argument("--num_features", type=int, default=8, help="Number of features per timestep")
    
    args = parser.parse_args()
    
    config_path = os.path.join(args.data_dir, "stations_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found at {config_path}. Please run data_loader.py first to generate the dataset.")
        
    with open(config_path, "r") as f:
        station_configs = json.load(f)
        
    station_ids = [c["station_id"] for c in station_configs]
    num_clients = len(station_ids)
    
    if num_clients == 0:
        raise ValueError("No clients found in the configuration.")
        
    print(f"Found {num_clients} clients (stations).")
    
    # Flower requires a function that spins up a client given its ID (string)
    def client_fn(cid: str) -> fl.client.Client:
        # We map cid (string integer from 0 to N-1) to our station_id
        station_id = station_ids[int(cid)]
        # return .to_client() wraps our NumPyClient in a standard Client interface for Flower
        client = AirQualityClient(
            station_id=station_id,
            data_dir=args.data_dir,
            model_name=args.model,
            window_size=args.window_size,
            num_features=args.num_features,
            epochs=1,
            batch_size=32
        )
        return client.to_client()
        
    # Minimum clients needed to start the simulation
    min_clients = max(1, int(args.fraction_fit * num_clients))
    
    strategy = get_strategy(
        strategy_name=args.strategy,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0, # Evaluate on all clients
        min_fit_clients=min_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        proximal_mu=args.proximal_mu
    )
    
    print(f"\n--- Starting {args.strategy.upper()} Simulation with {args.model.upper()} ---")
    
    # Start the simulation framework
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    # Extract logs to a local CSV
    metrics = []
    
    if history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            metrics.append({"round": rnd, "val_loss_mse": loss})
            
    if history.metrics_distributed and "rmse" in history.metrics_distributed:
        for rnd, rmse in history.metrics_distributed["rmse"]:
            # Insert the RMSE to the corresponding round dictionary
            for m in metrics:
                if m["round"] == rnd:
                    m["val_rmse"] = rmse
                    
    df_metrics = pd.DataFrame(metrics)
    log_file = f"metrics_{args.strategy}_{args.model}.csv"
    df_metrics.to_csv(log_file, index=False)
    
    print(f"\nSimulation complete. Per-round metrics saved to {log_file}")
    print(df_metrics.to_string(index=False))
    
    print("\n" + "="*50)
    print(" INSTRUCTIONS FOR GOOGLE COLAB")
    print("="*50)
    print("1. Upload this entire project folder to Google Drive.")
    print("2. Mount your drive in Colab:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("3. Install the required dependencies in a cell:")
    print("   !pip install flwr pandas numpy torch scikit-learn pyarrow fastparquet")
    print("4. Navigate to the project directory:")
    print("   %cd /content/drive/MyDrive/YOUR_PROJECT_FOLDER")
    print("5. Generate dataset (if not already done):")
    print("   !python data_loader.py")
    print(f"6. Run this script to execute the simulation:")
    print(f"   !python run_federated.py --strategy {args.strategy} --model {args.model}")
    print("="*50)

if __name__ == "__main__":
    main()
