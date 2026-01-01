# client_app.py
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import parameters_to_ndarrays, Parameters, NDArrays

from fedge.task import Net, load_data, set_weights, test, get_weights
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
        local_lr: float = 0.05,
    ):
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # SCAFFOLD needs its own optimizer + loss
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.local_lr)
        self.criterion = nn.CrossEntropyLoss()

    def _unwrap_parameters(
        self, parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Handle either a raw list of NDArrays or a Parameters object."""
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Any]
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train locally and return (diffs, num_examples, metrics)."""

        # 0) Start cost timer
        t_start = time.perf_counter()
        # 1) Unwrap & split
        return_diff = config.get("return_diff", False)
        n_layers   = config.get("n_layers", None)

        if return_diff and n_layers is not None:
            # server sent [global…, c_global…, c_local…]
            all_nd = self._unwrap_parameters(parameters)
            m     = int(n_layers)
            global_nd = all_nd[:m]
            c_global  = all_nd[m : 2*m]
            c_local   = all_nd[2*m : 3*m]
        else:
            # legacy: server put c_global / c_local in config
            global_nd = self._unwrap_parameters(parameters)
            c_global  = [np.array(x) for x in config.get("c_global", [])]
            c_local   = [np.array(x) for x in config.get("c_local", [])]

        # 2) Load into model
        set_weights(self.net, global_nd)

        # 3) Build name->array maps from state_dict order
        sd_keys = list(self.net.state_dict().keys())
        m = len(sd_keys)
        assert len(c_global) == m and len(c_local) == m, "control variates must match state_dict size"
        c_global_map = {k: v for k, v in zip(sd_keys, c_global)}
        c_local_map  = {k: v for k, v in zip(sd_keys, c_local)}

        # 4) Local SCAFFOLD‐corrected training
        loss_sum = 0.0
        correct = 0
        total_samples = 0
        step_count = 0  # Track number of optimizer steps
        
        def scaffold_ctrl_term() -> torch.Tensor:
            terms = []
            for name, p in self.net.named_parameters():  # trainable only, matched by name
                cg = torch.as_tensor(c_global_map[name], device=self.device, dtype=p.dtype)
                cl = torch.as_tensor(c_local_map[name],  device=self.device, dtype=p.dtype)
                terms.append((cg.reshape(-1) - cl.reshape(-1)).dot(p.reshape(-1)))
            return sum(terms) if terms else torch.zeros((), device=self.device)
        
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                if y.ndim > 1:
                    y = y.squeeze()
                y = y.long()

                self.optimizer.zero_grad()
                logits = self.net(x)
                ce_loss = self.criterion(logits, y)
                ctrl = scaffold_ctrl_term()
                loss = ce_loss + ctrl
                loss.backward()
                self.optimizer.step()
                step_count += 1  # Increment step count

                # accumulate training stats
                loss_sum += ce_loss.item() * y.size(0)   # ← multiply by batch size
                _, preds = torch.max(logits, 1)
                correct += (preds == y).sum().item()
                total_samples += y.size(0)

        # 4) Compute deltas
        local_nd = get_weights(self.net)
        # y_delta = W_local - W_global
        y_delta = [loc - glob for loc, glob in zip(local_nd, global_nd)]
        # c_local update: c_i^new = c_i - c_global - (y_delta / (η * K))
        K = max(1, step_count)  # Number of local optimizer steps
        c_local_new = [
            cl - cg - (yd / (self.local_lr * K))
            for cl, yd, cg in zip(c_local, y_delta, c_global)
        ]
        c_delta = [new - old for new, old in zip(c_local_new, c_local)]

        # 5) Compute cost metrics and prepare return
        out_nd = y_delta + c_delta
        
        # --- Cost calculations ---
        comp_time_sec = time.perf_counter() - t_start
        recv_arrays = list(global_nd) + list(c_global) + list(c_local)
        download_bytes = int(sum(arr.nbytes for arr in recv_arrays))
        upload_bytes   = int(sum(arr.nbytes for arr in out_nd))

        # --- Training effectiveness stats ---
        train_loss_mean = loss_sum / max(total_samples, 1)
        train_accuracy_mean = correct / max(total_samples, 1)

        # --- Update norms ---
        y_delta_norm = float(np.linalg.norm(np.concatenate([yd.ravel() for yd in y_delta])))
        c_delta_norm = float(np.linalg.norm(np.concatenate([cd.ravel() for cd in c_delta])))

        fit_metrics = {
            "comp_time_sec":   comp_time_sec,
            "download_bytes":  download_bytes,
            "upload_bytes":    upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "y_delta_norm":    y_delta_norm,
            "c_delta_norm":    c_delta_norm,
        }

        # Flower will wrap this tuple into FitRes
        return out_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        # 0) Cost timer and download bytes setup
        t_eval_start = time.perf_counter()
        # 1) Standard global‐model eval
        nd = self._unwrap_parameters(parameters)
        set_weights(self.net, nd)

        # compute download bytes for evaluation
        eval_download_bytes = int(sum(arr.nbytes for arr in nd))

        test_loss, test_acc   = test(self.net, self.valloader, self.device)
        train_loss, train_acc = test(self.net, self.trainloader, self.device)

        metrics = {
            "test_accuracy":  test_acc,
            "train_accuracy": train_acc,
            "accuracy_gap":   train_acc - test_acc,
            "test_loss":      test_loss,
            "train_loss":     train_loss,
            "loss_gap":       test_loss - train_loss,
        }

        # 2) Personalized eval: fine‐tune one more epoch on local data
        personal_net = copy.deepcopy(self.net)
        personal_net.to(self.device)
        p_opt = torch.optim.SGD(personal_net.parameters(), lr=self.local_lr)
        
        # (Or any personalization routine you prefer)
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            if y.ndim > 1:
                y = y.squeeze()
            y = y.long()
            p_opt.zero_grad()
            logits = personal_net(x)
            loss = self.criterion(logits, y)
            loss.backward()
            p_opt.step()

        p_test_loss, p_test_acc = test(personal_net, self.valloader, self.device)
        # 3) Relative improvement ratio
        rel_imp = (p_test_acc - test_acc) / (test_acc + 1e-12)

        metrics.update({
            "personalized_test_accuracy": p_test_acc,
            "personalized_test_loss":     p_test_loss,
            "relative_improvement":       rel_imp,
        })

        # add evaluation cost numbers
        metrics["comp_time_sec"] = time.perf_counter() - t_eval_start
        metrics["download_bytes"] = eval_download_bytes
        metrics["upload_bytes"] = len(pickle.dumps(metrics))

        return test_loss, len(self.valloader.dataset), metrics
    
def client_fn(context):
    # 1) Data + model
    batch_size = int(context.run_config.get("batch_size", 64))
    data_root = context.run_config.get("data-root", "hhar")
    use_watches = context.run_config.get("use-watches", True)
    sample_rate_hz = int(context.run_config.get("sample-rate-hz", 50))
    window_seconds = int(context.run_config.get("window-seconds", 2))
    window_stride_seconds = int(context.run_config.get("window-stride-seconds", 1))
    num_classes = int(context.run_config.get("num-classes", 6))
    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )

    trainloader, valloader, n_classes = load_data(
        context.node_config["partition-id"],      # keep: provided per node
        context.node_config["num-partitions"],    # keep: provided per node
        batch_size=batch_size,
        data_root=data_root,
        use_watches=use_watches,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        num_classes=num_classes,
    )

    sample, _ = next(iter(trainloader))          # (B, C, T)
    assert sample.ndim == 3, "HHAR tensors must be (B, C, T)"
    _, c, t = sample.shape
    net = Net(in_ch=c, seq_len=t, num_classes=n_classes)  # same class name as before

    # 2) Return NumPyClient
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
    ).to_client()

app = ClientApp(client_fn)
