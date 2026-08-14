from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np

class CalibrationAwareStrategy(fl.server.strategy.FedAvg):
    """
    Custom strategy that extends FedAvg (or FedProx theoretically, as proximal term is client-side).
    It adjusts the aggregation weights based on a 'calibration_score' provided by clients.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # Convert results to ndarrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics.get("calibration_score", 1.0))
            for _, fit_res in results
        ]

        # Calculate custom weights
        # Instead of weighting just by num_examples (FedAvg), we combine it with calibration_score
        # e.g., weight_i = num_examples_i * calibration_score_i
        
        total_weight = sum([num_examples * cal_score for _, num_examples, cal_score in weights_results])
        
        if total_weight == 0:
            # Fallback to standard FedAvg if scores are 0
            total_weight = sum([num_examples for _, num_examples, _ in weights_results])
            aggregated_ndarrays = [
                sum(layer) / total_weight
                for layer in zip(*[
                    [layer * num_examples for layer in weights]
                    for weights, num_examples, _ in weights_results
                ])
            ]
        else:
            # Calibration-aware aggregation
            aggregated_ndarrays = [
                sum(layer) / total_weight
                for layer in zip(*[
                    [layer * (num_examples * cal_score) for layer in weights]
                    for weights, num_examples, cal_score in weights_results
                ])
            ]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if needed
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            pass # print warning or handle default

        return parameters_aggregated, metrics_aggregated
