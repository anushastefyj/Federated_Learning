import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

class CalibrationAwareFedProx(fl.server.strategy.FedProx):
    """
    A custom strategy that extends FedProx by incorporating a calibration-aware 
    aggregation mechanism. It weights client updates based on their local validation 
    calibration error instead of solely on their number of training examples.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_calibration_stats = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
            
        # Extract weights and number of examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Extract calibration statistics returned by the clients via the metrics dict
        # We assume `val_mae` represents the magnitude of the calibration error
        client_maes = [fit_res.metrics.get("val_mae", 1.0) for _, fit_res in results]
        client_biases = [fit_res.metrics.get("val_mean_error", 0.0) for _, fit_res in results]
        
        # Log stats for this round
        avg_mae = float(np.mean(client_maes))
        avg_bias = float(np.mean(client_biases))
        
        self.round_calibration_stats.append({
            "round": server_round,
            "avg_mae": avg_mae,
            "avg_bias": avg_bias
        })
        
        # ----------------------------------------------------------------------
        # DESIGN CHOICE: Calibration Reweighting
        # Standard FedAvg weights clients purely by `num_examples`.
        # Here, we penalize clients that have poor calibration (high MAE).
        # We calculate the weight as: w = num_examples / (mae + epsilon)
        # This gives higher influence to clients whose local models are well-calibrated.
        #
        # EXPERIMENTATION POINT:
        # You can test different weighting schemes here. 
        # e.g., Exponential decay: w = num_examples * np.exp(-beta * mae)
        # e.g., Hard threshold: w = num_examples if mae < threshold else 0
        # ----------------------------------------------------------------------
        epsilon = 1e-5
        new_weights = []
        total_weight = 0.0
        
        for (weights, num_examples), mae in zip(weights_results, client_maes):
            w = num_examples / (mae + epsilon)
            new_weights.append((weights, w))
            total_weight += w
            
        # Manually aggregate the weights based on our new calibration scores
        aggregated_ndarrays = [
            np.zeros_like(layer) for layer in new_weights[0][0]
        ]
        
        for weights, w in new_weights:
            for i, layer in enumerate(weights):
                aggregated_ndarrays[i] += layer * (w / total_weight)
                
        # ----------------------------------------------------------------------
        # EXPERIMENTATION POINT: Bias Correction
        # We could also apply a global bias correction to the final output layer 
        # based on the weighted sum of `client_biases`. 
        # e.g., aggregated_ndarrays[-1] -= global_bias_correction
        # For simplicity and stability, we stick to trust-based reweighting above.
        # ----------------------------------------------------------------------
                
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Package and return aggregated metrics
        metrics = {
            "avg_calibration_mae": avg_mae,
            "avg_calibration_bias": avg_bias
        }
        
        return aggregated_parameters, metrics
