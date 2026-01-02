# client_app.py
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, NDArrays

from fedge.task import Net, load_data, set_weights, test, get_weights
import time  # For computation and comms cost tracking
import pickle  # For measuring size of metrics payload


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        local_epochs: int,
        local_lr: float = 0.05,
        client_id: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.client_id = client_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Randomness control for reproducibility (seed from config)
        torch.manual_seed(seed + client_id)
        np.random.seed(seed + client_id)

        # Store global model parameters for proximal term
        self.global_params = None

    def _unwrap_parameters(
        self, parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Handle either a raw list of NDArrays or a Parameters object."""
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """FedProx training with proximal term."""
        # Start cost timer
        t_start = time.perf_counter()
        
        # Unwrap and load global parameters
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)
        
        # Store global parameters for proximal term
        self.global_params = [param.clone().detach() for param in self.net.parameters()]
        
        # Get FedProx hyperparameters
        proximal_mu = config.get("proximal_mu", 0.01)
        local_epochs = config.get("local-epochs", self.local_epochs)
        
        # Create optimizer
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.local_lr)
        
        # Training metrics
        all_losses = []
        all_correct = 0
        all_total = 0
        
        # FedProx training
        self.net.train()
        for epoch in range(local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if y.ndim > 1:
                    y = y.squeeze()
                y = y.long()
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.net(x)
                loss = self.criterion(logits, y)
                
                # Add proximal term: μ/2 * ||w - w_global||²
                if proximal_mu > 0:
                    proximal_term = 0.0
                    for param, global_param in zip(self.net.parameters(), self.global_params):
                        proximal_term += torch.sum((param - global_param) ** 2)
                    loss += (proximal_mu / 2.0) * proximal_term
                
                # Backward pass and update
                loss.backward()
                optimizer.step()
                
                # Accumulate stats
                all_losses.append(loss.item())
                _, preds = torch.max(logits, 1)
                all_correct += (preds == y).sum().item()
                all_total += y.size(0)
        
        # Prepare results
        local_nd = get_weights(self.net)
        t_end = time.perf_counter()
        
        download_bytes = int(sum(arr.nbytes for arr in global_nd))
        upload_bytes = int(sum(arr.nbytes for arr in local_nd))
        self.comp_time_sec = t_end - t_start
        self.upload_bytes = upload_bytes
        self.download_bytes = download_bytes
        
        # Compute training metrics
        train_loss_mean = np.mean(all_losses) if all_losses else 0.0
        train_accuracy_mean = all_correct / max(1, all_total) if all_total > 0 else 0.0
        
        fit_metrics = {
            "comp_time_sec": self.comp_time_sec,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "total_train_samples": all_total,
        }
        return local_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Evaluate the global model."""
        t_eval_start = time.perf_counter()
        
        # Load global parameters and evaluate
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)
        eval_download_bytes = int(sum(arr.nbytes for arr in global_nd))
        
        test_loss, test_acc = test(self.net, self.valloader, self.device)
        train_loss, train_acc = test(self.net, self.trainloader, self.device)
        
        metrics = {
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "accuracy_gap": train_acc - test_acc,
            "test_loss": test_loss,
            "train_loss": train_loss,
            "loss_gap": test_loss - train_loss,
            "comp_time_sec": getattr(self, "comp_time_sec", time.perf_counter() - t_eval_start),
            "download_bytes": eval_download_bytes,
        }
        
        metrics_payload_bytes = len(pickle.dumps(metrics))
        metrics["upload_bytes"] = getattr(self, "upload_bytes", 0) + metrics_payload_bytes
        
        return test_loss, len(self.valloader.dataset), metrics
    
def client_fn(context):
    # 1) Data + model using HHAR parameters
    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )
    
    # Load HHAR data with parameters from run_config
    trainloader, valloader, n_classes = load_data(
        partition_id=pid,
        num_partitions=num_parts,
        batch_size=context.run_config.get("batch_size", 32),
        data_root=context.run_config.get("data-root", "hhar"),
        use_watches=context.run_config.get("use-watches", True),
        sample_rate_hz=context.run_config.get("sample-rate-hz", 50),
        window_seconds=context.run_config.get("window-seconds", 2),
        window_stride_seconds=context.run_config.get("window-stride-seconds", 1),
        num_classes=context.run_config.get("num-classes", 6),
    )
    
    # Infer model shape from HHAR data: (B, C, T)
    sample, _ = next(iter(trainloader))
    _, c, t = sample.shape
    net = Net(in_ch=c, seq_len=t, num_classes=n_classes)

    # 2) Return NumPyClient
    # Priority: ENV variable SEED > config file > default (42)
    seed = int(os.environ.get("SEED", context.run_config.get("seed", 42)))
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
        client_id=pid,
        seed=seed,
    ).to_client()

app = ClientApp(client_fn)
