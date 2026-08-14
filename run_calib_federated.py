import os
import json
import argparse
import pandas as pd
import flwr as fl

from fl.fl_client import AirQualityClient
from fl.calibration_strategy import CalibrationAwareFedProx

def main():
    parser = argparse.ArgumentParser(description="Calibration-Aware FL Simulation")
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Directory with processed station data")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"], help="Model architecture")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients selected for fit")
    parser.add_argument("--proximal_mu", type=float, default=0.1, help="Proximal term for FedProx")
    parser.add_argument("--window_size", type=int, default=12, help="Time window size")
    parser.add_argument("--num_features", type=int, default=8, help="Number of features per timestep")
    
    args = parser.parse_args()
    
    config_path = os.path.join(args.data_dir, "stations_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found at {config_path}.")
        
    with open(config_path, "r") as f:
        station_configs = json.load(f)
        
    station_ids = [c["station_id"] for c in station_configs]
    num_clients = len(station_ids)
    
    if num_clients == 0:
        raise ValueError("No clients found in the configuration.")
        
    print(f"Found {num_clients} clients (stations).")
    
    def client_fn(cid: str) -> fl.client.Client:
        station_id = station_ids[int(cid)]
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
        
    min_clients = max(1, int(args.fraction_fit * num_clients))
    
    def evaluate_metrics_aggregation_fn(eval_metrics):
        if not eval_metrics:
            return {}
        total_samples = sum([n for n, _ in eval_metrics])
        weighted_rmse = sum([n * m["rmse"] for n, m in eval_metrics]) / total_samples
        return {"rmse": weighted_rmse}

    # Instantiate our custom Calibration-Aware strategy
    strategy = CalibrationAwareFedProx(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0, 
        min_fit_clients=min_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        proximal_mu=args.proximal_mu,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    
    print(f"\n--- Starting CALIBRATION-AWARE FEDPROX Simulation with {args.model.upper()} ---")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    # 1. Log Standard FL Metrics (Loss & RMSE)
    metrics = []
    if history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            metrics.append({"round": rnd, "val_loss_mse": loss})
            
    if history.metrics_distributed and "rmse" in history.metrics_distributed:
        for rnd, rmse in history.metrics_distributed["rmse"]:
            for m in metrics:
                if m["round"] == rnd:
                    m["val_rmse"] = rmse
                    
    # 2. Log Calibration Stats injected via custom strategy
    # The custom strategy returns avg_calibration_mae and avg_calibration_bias inside aggregate_fit 
    # which Flower captures inside history.metrics_distributed
    if history.metrics_distributed:
        if "avg_calibration_mae" in history.metrics_distributed:
            for rnd, mae in history.metrics_distributed["avg_calibration_mae"]:
                for m in metrics:
                    if m["round"] == rnd: m["avg_calibration_mae"] = mae
                    
        if "avg_calibration_bias" in history.metrics_distributed:
            for rnd, bias in history.metrics_distributed["avg_calibration_bias"]:
                for m in metrics:
                    if m["round"] == rnd: m["avg_calibration_bias"] = bias
                    
    df_metrics = pd.DataFrame(metrics)
    log_file = f"metrics_calib_fedprox_{args.model}.csv"
    df_metrics.to_csv(log_file, index=False)
    
    print(f"\nSimulation complete. Per-round metrics saved to {log_file}")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
