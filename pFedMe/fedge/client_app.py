# client_app.py
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, NDArrays

from fedge.task import Net, load_data, set_weights, test, get_weights, DATA_FLAGS
import copy
import time  # For computation and comms cost tracking
import pickle  # For measuring size of metrics payload


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        local_epochs: int,
        local_lr: float = 1e-2,
        client_id: int = 0,
    ):
        super().__init__()
        self.net = net  # This will be the w model (global model copy)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.client_id = client_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Create personalized model theta (separate from w)
        self.theta_net = copy.deepcopy(self.net)
        self.theta_net.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Randomness control for reproducibility
        torch.manual_seed(42 + client_id)
        np.random.seed(42 + client_id)
        
        # Initialize theta as copy of initial global model
        self.theta_initialized = False

    def _unwrap_parameters(
        self, parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Handle either a raw list of NDArrays or a Parameters object."""
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Authentic pFedMe bi-level optimization with inner/outer loops."""
        # Start cost timer
        t_start = time.perf_counter()
        
        # Unwrap and load global parameters into w model
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)  # w ← global model
        
        # Canonical pFedMe: reset θ from the current global w each round
        set_weights(self.theta_net, global_nd)  # θ ← w (reset every round)
        self.theta_initialized = True
        
        # Get pFedMe hyperparameters
        lamda = config.get("lamda", 15.0)
        inner_steps = config.get("inner_steps", 5)
        outer_steps = config.get("outer_steps", 1)
        inner_lr = config.get("inner_lr", 0.01)
        outer_lr = config.get("outer_lr", 0.01)
        local_epochs = config.get("local-epochs", self.local_epochs)
        
        # Canonical pFedMe: use lambda directly (no param-count scaling) and precompute once
        lamda_eff = float(lamda)
        
        # Create optimizers
        theta_optimizer = torch.optim.SGD(self.theta_net.parameters(), lr=inner_lr)
        
        # Improved train metrics - track all inner batches
        all_losses = []
        all_correct = 0
        all_total = 0
        
        # pFedMe bi-level optimization
        for epoch in range(local_epochs):
            # Outer loop (R steps) - typically R=1
            for outer_step in range(outer_steps):
                
                # Inner loop (K mini-batches) - optimize θ
                batch_count = 0
                for x, y in self.trainloader:
                    if batch_count >= inner_steps:
                        break
                        
                    x, y = x.to(self.device), y.to(self.device)
                    if y.ndim > 1:
                        y = y.squeeze()
                    y = y.long()
                    
                    # Zero gradients for theta optimization
                    theta_optimizer.zero_grad()
                    
                    # Forward pass with theta model
                    logits = self.theta_net(x)
                    loss = self.criterion(logits, y)

                    # Add Moreau envelope regularization: λ_eff/2 * sum(||θ - w||²)
                    reg = torch.tensor(0.0, device=self.device)
                    for theta_param, w_param in zip(self.theta_net.parameters(), self.net.parameters()):
                        diff = theta_param - w_param.detach()
                        reg = reg + torch.sum(diff * diff)
                    loss = loss + (lamda_eff / 2.0) * reg

                    # (optional but strong) guard before backward
                    if not torch.isfinite(loss).all():
                        set_weights(self.theta_net, get_weights(self.net))  # θ ← w
                        break

                    # Backward pass and update θ
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), 1.0)
                    theta_optimizer.step()

                    
                    # Accumulate stats from all inner batches for better metrics
                    all_losses.append(loss.item())
                    _, preds = torch.max(logits, 1)
                    all_correct += (preds == y).sum().item()
                    all_total += y.size(0)
                    
                    batch_count += 1
                
                # Outer step: update w ← w - η*λ*(w - θ)
                with torch.no_grad():
                    for w_param, theta_param in zip(self.net.parameters(), self.theta_net.parameters()):
                        w_param.data = w_param.data - outer_lr * lamda_eff  * (w_param.data - theta_param.data)
        
        # Prepare results
        local_nd = get_weights(self.net)  # Return updated w
        t_end = time.perf_counter()
        
        download_bytes = int(sum(arr.nbytes for arr in global_nd))
        upload_bytes = int(sum(arr.nbytes for arr in local_nd))
        self.comp_time_sec = t_end - t_start
        self.upload_bytes = upload_bytes
        self.download_bytes = download_bytes
        
        # Compute training metrics from all inner batches
        train_loss_mean = np.mean(all_losses) if all_losses else 0.0
        train_accuracy_mean = all_correct / max(1, all_total) if all_total > 0 else 0.0
        
        fit_metrics = {
            "cid": int(self.client_id),                    # NEW
            "comp_time_sec": self.comp_time_sec,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "num_inner_batches": len(all_losses),
            "total_train_samples": all_total,
            # SCAFFOLD placeholders to match template shape
            "y_delta_norm": 0.0,
            "c_delta_norm": 0.0,
        }
        return local_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Evaluate global and personalized models."""
        t_eval_start = time.perf_counter()
        # Global model evaluation
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
        }
        # Personalized model evaluation using theta_net
        # Always initialize theta_net if not already done
        if not self.theta_initialized:
            set_weights(self.theta_net, global_nd)  # θ ← w (initialize with global model)
            self.theta_initialized = True
            
        p_test_loss, p_test_acc = test(self.theta_net, self.valloader, self.device)
        rel_improvement = (p_test_acc - test_acc) / (test_acc + 1e-12)
        
        # Debug personalization effectiveness
        if abs(rel_improvement) > 1e-6:  # Only log if there's meaningful improvement
            print(f"Client {self.client_id}: Global acc={test_acc:.6f}, Personal acc={p_test_acc:.6f}, Improvement={rel_improvement:.6f}")
        
        metrics.update({
            "personalized_test_accuracy": p_test_acc,
            "personalized_test_loss": p_test_loss,
            "relative_improvement": rel_improvement,
        })
        metrics["comp_time_sec"] = getattr(self, "comp_time_sec", time.perf_counter() - t_eval_start)
        metrics["download_bytes"] = eval_download_bytes
        metrics["upload_bytes"] = getattr(self, "upload_bytes", 0) + len(pickle.dumps(metrics))
        metrics["cid"] = int(self.client_id)   # NEW
        return test_loss, len(self.valloader.dataset), metrics
    
def client_fn(context):
    # 1) Data + model
    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )
    
    # Use Scaffold-compatible load_data signature
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
    sample, _ = next(iter(trainloader))
    if sample.ndim == 4:
        _, c, h, w = sample.shape
        net = Net(in_ch=c, img_h=h, img_w=w, n_class=n_classes)
    elif sample.ndim == 3:
        _, c, t = sample.shape
        net = Net(in_ch=c, n_class=n_classes, seq_len=t)
    else:
        raise ValueError(f"Unsupported input shape: {tuple(sample.shape)}")

    # 2) Return NumPyClient
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
        client_id=pid,  # For reproducibility
    ).to_client()

app = ClientApp(client_fn)
