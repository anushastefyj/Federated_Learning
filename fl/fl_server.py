import flwr as fl
from typing import Dict, List, Optional, Tuple

def get_strategy(
    strategy_name="fedavg",
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    proximal_mu=0.0
):
    """
    Returns a Flower Strategy based on configuration.
    
    # -------------------------------------------------------------
    # CALIBRATION-AWARE AGGREGATION EXTENSION POINT
    # -------------------------------------------------------------
    # To plug in the calibration-aware aggregation later, you will:
    # 1. Subclass flwr.server.strategy.FedAvg (or FedProx).
    # 2. Override the `aggregate_fit` method.
    # 3. Inside `aggregate_fit`, re-weight the received `results` 
    #    (list of weights from clients) using your calibration 
    #    scores (e.g., trust scores or spatial reliability metrics) 
    #    instead of standard sample-size based weighting.
    # -------------------------------------------------------------
    """
    
    def evaluate_metrics_aggregation_fn(eval_metrics):
        """Aggregates metrics (like RMSE) returned by clients during evaluate phase."""
        if not eval_metrics:
            return {}
        
        # We perform a weighted average of the RMSE scores based on number of samples evaluated
        total_samples = sum([n for n, _ in eval_metrics])
        weighted_rmse = sum([n * m["rmse"] for n, m in eval_metrics]) / total_samples
        return {"rmse": weighted_rmse}

    if strategy_name.lower() == "fedavg":
        return fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
    elif strategy_name.lower() == "fedprox":
        return fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            proximal_mu=proximal_mu,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
