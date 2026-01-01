import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional, Union

from flwr.common import (
    FitIns, FitRes, NDArrays, Parameters, 
    ndarrays_to_parameters, parameters_to_ndarrays,
    Scalar
)
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class pFedMe(FedAvg):
    """
    Authentic pFedMe strategy implementing bi-level optimization.

    Algorithm:
    1. Clients receive global model w.
    2. Clients perform K inner steps optimizing θ with Moreau envelope.
    3. Clients perform R outer steps updating w toward θ.
    4. Server aggregates w with optional β-mixing.
    """

    def __init__(
        self,
        lamda: float = 15.0,
        inner_steps: int = 5,
        outer_steps: int = 1,
        inner_lr: float = 0.01,
        outer_lr: float = 0.01,
        beta: float = 1.0,  # Server mixing parameter
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.lamda = lamda
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.beta = beta
        
        # Store previous global model for β-mixing
        self.previous_global_parameters: Optional[NDArrays] = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Cache previous global parameters for β-mixing (except round 1)
        if server_round > 1 and parameters is not None:
            self.previous_global_parameters = parameters_to_ndarrays(parameters)
        
        # Use FedAvg to select clients and get FitIns
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        customized: List[Tuple[ClientProxy, FitIns]] = []
        for client, ins in fit_ins:
            # Inject pFedMe hyperparameters into config
            config = {
                **ins.config, 
                "lamda": self.lamda,
                "inner_steps": self.inner_steps,
                "outer_steps": self.outer_steps,
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
            }
            customized.append((client, FitIns(parameters=ins.parameters, config=config)))
        return customized
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates with pFedMe-specific β-mixing and invoke metrics aggregator."""
        if not results:
            return None, {}

        # 1) Weighted average of client weights (standard FedAvg)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        weighted_avg = self._weighted_average(weights_results)

        # 2) β-mixing with previous global
        if self.previous_global_parameters is not None:
            mixed_parameters = self._beta_mixing(
                weighted_avg, self.previous_global_parameters, self.beta
            )
        else:
            mixed_parameters = weighted_avg

        # 3) Call configured fit-metrics aggregator to populate server cache
        metrics_aggregated: Dict[str, Scalar] = {}
        agg_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if agg_fn is not None:
            try:
                # Newer style: (server_round, results, failures)
                metrics_aggregated = agg_fn(server_round, results, failures)
            except TypeError:
                # Classic style: list of (num_examples, metrics)
                metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
                metrics_aggregated = agg_fn(metrics_list)

        # 4) Update previous_global and return
        self.previous_global_parameters = mixed_parameters
        return ndarrays_to_parameters(mixed_parameters), metrics_aggregated
    
    def _uniform_average(self, weights_list: List[NDArrays]) -> NDArrays:
        """Average params from multiple clients; preserve original dtypes."""
        num_clients = len(weights_list)
        if num_clients == 0:
            return []
        ref_dtypes = [w.dtype for w in weights_list[0]]

        # Accumulate in float64 to avoid integer casting issues
        acc = [w.astype(np.float64, copy=True) for w in weights_list[0]]
        for other in weights_list[1:]:
            for i, w in enumerate(other):
                acc[i] += w.astype(np.float64, copy=False)

        # Divide and cast back to original dtype
        for i in range(len(acc)):
            acc[i] /= float(num_clients)
            if np.issubdtype(ref_dtypes[i], np.floating):
                acc[i] = acc[i].astype(ref_dtypes[i], copy=False)
            else:
                acc[i] = np.rint(acc[i]).astype(ref_dtypes[i], copy=False)
        return acc

    def _weighted_average(self, weights_results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Weighted average preserving original dtypes."""
        if not weights_results:
            return []
        total_weight = float(sum(w for _, w in weights_results))
        ref_dtypes = [w.dtype for w in weights_results[0][0]]

        acc = [np.zeros_like(w, dtype=np.float64) for w in weights_results[0][0]]
        for params, weight in weights_results:
            scale = float(weight) / total_weight
            for i, w in enumerate(params):
                acc[i] += w.astype(np.float64, copy=False) * scale

        for i in range(len(acc)):
            if np.issubdtype(ref_dtypes[i], np.floating):
                acc[i] = acc[i].astype(ref_dtypes[i], copy=False)
            else:
                acc[i] = np.rint(acc[i]).astype(ref_dtypes[i], copy=False)
        return acc
    
    def _beta_mixing(
        self, 
        new_params: NDArrays, 
        old_params: NDArrays, 
        beta: float
    ) -> NDArrays:
        """w^{t+1} = (1-β)·w^t + β·new_avg, preserving dtypes."""
        mixed: NDArrays = []
        for new_p, old_p in zip(new_params, old_params):
            m = (1.0 - beta) * old_p.astype(np.float64) + beta * new_p.astype(np.float64)
            # Cast back to the old param's dtype
            if np.issubdtype(old_p.dtype, np.floating):
                m = m.astype(old_p.dtype, copy=False)
            else:
                m = np.rint(m).astype(old_p.dtype, copy=False)
            mixed.append(m)
        return mixed
