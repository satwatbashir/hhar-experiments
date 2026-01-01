# client_app.py
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, NDArrays

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

        cid = config.get("cid")
        cluster_id = int(config.get("cluster_id", -1))
        # 0) Start cost timer
        t_start = time.perf_counter()

        # 1) Determine training mode -------------------------------------------------
        return_diff = config.get("return_diff", False)
        n_layers    = config.get("n_layers")  # may be None

        scaffold_mode = return_diff and n_layers is not None

        if scaffold_mode:
            # Server provided [global …, c_global …, c_local …]
            all_nd = self._unwrap_parameters(parameters)
            m = int(n_layers)
            global_nd = all_nd[:m]
            c_global  = all_nd[m : 2 * m]
            c_local   = all_nd[2 * m : 3 * m]
        else:
            # CFL / FedAvg: only weights were sent
            global_nd = self._unwrap_parameters(parameters)
            c_global, c_local = [], []

        # 2) Load sent weights into the local model
        set_weights(self.net, global_nd)

        # 3) Local training
        loss_sum = 0.0
        correct = 0
        total_samples = 0
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                if y.ndim > 1:
                    y = y.squeeze(-1)   # only drop the last dim, keep batch
                if y.ndim == 0:
                    y = y.unsqueeze(0)  # absolute safety for singleton cases
                y = y.long()

                self.optimizer.zero_grad()
                logits = self.net(x)
                loss = self.criterion(logits, y)

                if scaffold_mode:
                    # SCAFFOLD control-variate penalty term
                    ctrl = sum(
                        (
                            torch.tensor(cg, device=self.device) - torch.tensor(cl, device=self.device)
                        )
                        .reshape(-1)
                        .dot(p.reshape(-1))
                        for p, cg, cl in zip(self.net.parameters(), c_global, c_local)
                    )
                    (loss + ctrl).backward()
                else:
                    loss.backward()
                self.optimizer.step()

                # accumulate training stats
                loss_sum += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == y).sum().item()
                total_samples += y.size(0)

        # 4) Build return payload ----------------------------------------------
        local_nd = get_weights(self.net)
        if scaffold_mode:
            # y_delta = W_local – W_global
            y_delta = [loc - glob for loc, glob in zip(local_nd, global_nd)]
            # Update control-variates: cl_new = cl + (y_delta / (ηN)) – c_global
            N = float(len(self.trainloader.dataset))
            c_local_new = [
                cl + (yd / (self.local_lr * N)) - cg
                for cl, yd, cg in zip(c_local, y_delta, c_global)
            ]
            c_delta = [new - old for new, old in zip(c_local_new, c_local)]
            out_nd = y_delta + c_delta
        else:
            # CFL/FedAvg: send diff = weights_sent – weights_after_training
            out_nd = [glob - loc for glob, loc in zip(global_nd, local_nd)]
        # --- Cost calculations ---
        t_end = time.perf_counter()
        download_arrays = list(global_nd) + c_global + c_local
        upload_arrays   = list(out_nd)
        self.comp_time_sec  = t_end - t_start
        self.download_bytes = int(sum(arr.nbytes for arr in download_arrays))
        self.upload_bytes   = int(sum(arr.nbytes for arr in upload_arrays))

        # --- Training effectiveness stats ---
        train_loss_mean = loss_sum / max(len(self.trainloader) * self.local_epochs, 1)
        train_accuracy_mean = correct / total_samples if total_samples > 0 else 0.0

        # --- Prepare metrics ---------------------------------------------------
        fit_metrics = {
            "comp_time_fit_sec": self.comp_time_sec,
            "download_bytes_fit":  self.download_bytes,
            "upload_bytes_fit":    self.upload_bytes,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "num_train_samples": len(self.trainloader.dataset),
        }

        if scaffold_mode:
            y_delta_norm = float(np.linalg.norm(np.concatenate([yd.ravel() for yd in y_delta]))) if y_delta else 0.0
            c_delta_norm = float(np.linalg.norm(np.concatenate([cd.ravel() for cd in c_delta]))) if c_delta else 0.0
            fit_metrics.update({
                "y_delta_norm": y_delta_norm,
                "c_delta_norm": c_delta_norm,
            })

        # Flower will wrap this tuple into FitRes
        fit_metrics["cid"] = cid
        fit_metrics["cluster_id"] = cluster_id
        return out_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        cid = config.get("cid")
        cluster_id = int(config.get("cluster_id", -1))
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
            "num_test_samples": len(self.valloader.dataset),
        }

        # 2) Personalized eval: fine‐tune one more epoch on local data
        personal_net = copy.deepcopy(self.net)
        personal_net.to(self.device)
        personal_optimizer = torch.optim.SGD(personal_net.parameters(), lr=self.local_lr)
        # (Or any personalization routine you prefer)
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            if y.ndim > 1:
                y = y.squeeze(-1)   # only drop the last dim, keep batch
            if y.ndim == 0:
                y = y.unsqueeze(0)  # absolute safety for singleton cases
            y = y.long()
            personal_net.zero_grad()
            logits = personal_net(x)
            loss = self.criterion(logits, y)
            loss.backward()
            personal_optimizer.step()

        p_test_loss, p_test_acc = test(personal_net, self.valloader, self.device)
        t_eval_end = time.perf_counter()
        
        # 3) Relative improvement ratio
        rel_imp = (p_test_acc - test_acc) / (test_acc + 1e-12)

        metrics.update({
            "personalized_test_accuracy": p_test_acc,
            "personalized_test_loss":     p_test_loss,
            "relative_improvement":       rel_imp,
            "comp_time_sec":  getattr(self, "comp_time_sec", time.perf_counter() - t_eval_start),
            "comp_time_eval_sec": t_eval_end - t_eval_start,
            "download_bytes_eval": eval_download_bytes,
        })
        # Finalise upload bytes: size of metrics payload itself
        metrics_payload_bytes = len(pickle.dumps(metrics))
        metrics["upload_bytes_eval"] = metrics_payload_bytes

        metrics["cid"] = cid
        metrics["cluster_id"] = cluster_id
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

    from fedge.task import load_data, Net  # ensure we use the new task.py
    trainloader, valloader, n_classes = load_data(
        pid, num_parts,
        batch_size=batch_size,
        data_root=data_root,
        use_watches=use_watches,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        num_classes=num_classes,
    )

    sample, _ = next(iter(trainloader))  # (B, C, T)
    assert sample.ndim == 3, "HHAR tensors must be (B, C, T)"
    _, c, t = sample.shape
    net = Net(in_ch=c, seq_len=t, num_classes=n_classes)

    # 2) Return NumPyClient
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
    ).to_client()

app = ClientApp(client_fn)
