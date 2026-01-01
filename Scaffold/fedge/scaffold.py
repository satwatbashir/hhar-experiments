import os
import numpy as np
import torch
import time
import csv
from typing import Any, Dict, List, Tuple
from copy import deepcopy

from flwr.common import NDArrays, Parameters, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class Scaffold(FedAvg):
    """
    SCAFFOLD strategy for Flower.

    1. Server keeps a global control variate c_global.
    2. Each client keeps its own c_local.
    3. Clients solve local problem with gradient correction ∇F + c_global - c_local.
    4. Clients return two diffs: y_delta = W_local - W_global, and c_delta = c_local_new - c_local_old.
    5. Server updates:
         W_global ← W_global + η * average(y_delta)
         c_global ← c_global + average(c_delta)
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        global_lr: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.global_lr = global_lr
        self.c_global: NDArrays = []
        self.c_locals: Dict[str, NDArrays] = {}
        self._round_t0 = None  # wall-clock per round
        self._initialized = False
        # Store latest global parameters for update
        self._latest_global_parameters: Parameters | None = None

    def _init_control(self, parameters: Parameters) -> None:
        """Initialize c_global (zeros) on first round."""
        nd = parameters_to_ndarrays(parameters)
        self.c_global = [np.zeros(w.shape, dtype=np.float32) for w in nd]
        self._initialized = True

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # 0) Save the global parameters for later in aggregate_fit
        self._latest_global_parameters = parameters

        # 1) initialize c_global on round 0
        if not self._initialized:
            self._init_control(parameters)

        # 2) get the default FedAvg FitIns
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        custom: List[Tuple[ClientProxy, FitIns]] = []
        m = len(self.c_global)

        for client, ins in fit_ins:
            cid = client.cid
            if cid not in self.c_locals:
                self.c_locals[cid] = deepcopy(self.c_global)

            # pack global + c_global + c_local into one Parameters
            global_nd = parameters_to_ndarrays(ins.parameters)
            to_send = (
                global_nd
                + [w.copy() for w in self.c_global]
                + [w.copy() for w in self.c_locals[cid]]
            )
            send_params = ndarrays_to_parameters(to_send)

            cfg = {**ins.config, "return_diff": True, "n_layers": m}
            custom.append((client, FitIns(parameters=send_params, config=cfg)))

        self._round_t0 = time.perf_counter()
        return custom


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        # Buffers for the two kinds of deltas
        y_deltas: List[List[torch.Tensor]] = []
        c_deltas: List[List[torch.Tensor]] = []
        weights: List[int] = []

        # Per-client tracking
        per_client_rows = []
        bytes_down_total = 0
        bytes_up_total = 0

        m = len(self.c_global)

        # 1) Unpack each client's FitRes
        for client, fit_res in results:
            # Convert the returned Parameters → list of np.ndarray
            returned_params: Parameters = fit_res.parameters
            nd_list = parameters_to_ndarrays(returned_params)
            num_examples = fit_res.num_examples
            # metrics = fit_res.metrics  # if you need them

            # First m arrays are y_delta, next m are c_delta
            y_np = nd_list[:m]
            c_np = nd_list[m : 2 * m]

            y_deltas.append([torch.tensor(x) for x in y_np])
            c_deltas.append([torch.tensor(x) for x in c_np])
            weights.append(num_examples)

            # Capture per-client metrics and communications
            met = fit_res.metrics or {}

            # (1) Sum comms across clients
            up   = int(met.get("upload_bytes", 0))
            down = int(met.get("download_bytes", 0))
            bytes_up_total   += up
            bytes_down_total += down

            # (2) Store one row per client/round
            per_client_rows.append({
                "round": server_round,
                "cid": client.cid,
                # Backward-compat alias expected by some plotters
                "client_id": client.cid,
                "num_examples": int(fit_res.num_examples),
                "train_loss_mean": float(met.get("train_loss_mean")) if "train_loss_mean" in met else None,
                "train_accuracy_mean": float(met.get("train_accuracy_mean")) if "train_accuracy_mean" in met else None,
                "upload_bytes": up,
                "download_bytes": down,
                "comp_time_sec": float(met.get("comp_time_sec")) if "comp_time_sec" in met else None,
                "y_delta_norm": float(met.get("y_delta_norm")) if "y_delta_norm" in met else None,
                "c_delta_norm": float(met.get("c_delta_norm")) if "c_delta_norm" in met else None,
            })

            # Update this client's c_local with float32 operations
            cid = client.cid
            self.c_locals[cid] = [
                cl.astype(np.float32, copy=False) + cd.numpy().astype(np.float32)
                for cl, cd in zip(self.c_locals[cid], c_deltas[-1])
            ]

        # 2) Compute weighted average of y_deltas → global update
        total_weight = float(sum(weights))
        norm_weights = [w / total_weight for w in weights]

        # Fetch the previous global as a list
        prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
        W = [torch.tensor(x) for x in prev_global_nd]

        for idx, parts in enumerate(zip(*y_deltas)):
            stacked = torch.stack(list(parts), dim=0)  # shape: [num_clients, …]
            # build a broadcastable weight tensor
            w = torch.tensor(norm_weights, device=stacked.device)
            for _ in range(stacked.ndim - 1):
                w = w.unsqueeze(1)  # now shape [num_clients,1,1,…,1]
            avg_delta = (stacked * w).sum(dim=0)
            if W[idx].is_floating_point():
                W[idx] = W[idx] + self.global_lr * avg_delta
            # else: leave integer/bool buffers unchanged

        new_global = ndarrays_to_parameters([w.numpy() for w in W])

        # 3) Update c_global = c_global + avg(c_deltas)
    
        for idx, parts in enumerate(zip(*c_deltas)):
            stacked = torch.stack(list(parts), dim=0)
            w = torch.tensor(norm_weights, device=stacked.device)
            for _ in range(stacked.ndim - 1):
                w = w.unsqueeze(1)
            avg_c = (stacked * w).sum(dim=0)
            self.c_global[idx] = self.c_global[idx].astype(np.float32, copy=False) \
                                 + avg_c.cpu().numpy().astype(np.float32)

        # Wall clock for this round
        wall_clock_sec = None
        if self._round_t0 is not None:
            wall_clock_sec = float(time.perf_counter() - self._round_t0)

        # ---- Write per-client fit rows (RENAMED FILE) ----
        os.makedirs("metrics", exist_ok=True)
        clients_csv = os.path.join("metrics", "clients.csv")  # was client_fit_metrics.csv
        write_header = not os.path.exists(clients_csv)
        with open(clients_csv, "a", newline="") as f:
            keys = [
                "round","cid","client_id","num_examples",
                "train_loss_mean","train_accuracy_mean",
                "upload_bytes","download_bytes","comp_time_sec",
                "y_delta_norm","c_delta_norm",
            ]
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header: w.writeheader()
            for row in per_client_rows:
                w.writerow(row)

        # ---- Compute aggregated means for this round (fit) ----
        # Use client metrics captured earlier; weight by num_examples
        def _weighted_mean(key: str) -> float | None:
            vals, wts = [], []
            for row in per_client_rows:
                v = row.get(key)
                if v is None:
                    continue
                vals.append(float(v))
                wts.append(int(row["num_examples"]))
            if not vals:
                return None
            tot = float(sum(wts))
            return float(sum(v * (w/tot) for v, w in zip(vals, wts)))

        fit_train_loss_mean     = _weighted_mean("train_loss_mean")
        fit_train_accuracy_mean = _weighted_mean("train_accuracy_mean")
        comp_time_sec_mean      = _weighted_mean("comp_time_sec")

        # ---- Write one round row (NEW FILE rounds.csv) ----
        rounds_csv = os.path.join("metrics", "rounds.csv")
        write_header = not os.path.exists(rounds_csv)
        with open(rounds_csv, "a", newline="") as f:
            keys = ["round",
                    "fit_train_loss_mean","fit_train_accuracy_mean",
                    "bytes_up_total","bytes_down_total",
                    "comp_time_sec_mean","wall_clock_sec"]
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
            w.writerow({
                "round": server_round,
                "fit_train_loss_mean": fit_train_loss_mean if fit_train_loss_mean is not None else "",
                "fit_train_accuracy_mean": fit_train_accuracy_mean if fit_train_accuracy_mean is not None else "",
                "bytes_up_total": int(bytes_up_total),
                "bytes_down_total": int(bytes_down_total),
                "comp_time_sec_mean": comp_time_sec_mean if comp_time_sec_mean is not None else "",
                "wall_clock_sec": wall_clock_sec if wall_clock_sec is not None else "",
            })

        # Let FedAvg compute and return aggregated metrics (no CSV here)
        _, agg_metrics = super().aggregate_fit(server_round, results, failures)
        return new_global, dict(agg_metrics)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Aggregate evaluation results and write per-client test metrics to clients.csv."""
        
        # Per-client test metrics tracking
        per_client_test_rows = []
        
        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            metrics = eval_res.metrics or {}
            
            # Extract test metrics, fallback to eval_res.loss if needed
            test_loss = metrics.get("test_loss", eval_res.loss)
            test_accuracy = metrics.get("test_accuracy", metrics.get("test_acc", None))
            
            # Pull optional gaps and train metrics reported by clients
            acc_gap  = metrics.get("accuracy_gap", None)
            loss_gap = metrics.get("loss_gap", None)
            train_loss = metrics.get("train_loss", None) or metrics.get("train_loss", None)
            train_acc  = metrics.get("train_accuracy", None) or metrics.get("train_accuracy", None)

            if test_loss is not None:
                per_client_test_rows.append({
                    "round": server_round,
                    "cid": cid,
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
                    "train_loss": float(train_loss) if train_loss is not None else "",
                    "train_accuracy": float(train_acc) if train_acc is not None else "",
                    "accuracy_gap": float(acc_gap) if acc_gap is not None else "",
                    "loss_gap": float(loss_gap) if loss_gap is not None else "",
                    "personalized_test_accuracy": float(metrics.get("personalized_test_accuracy", "")) if "personalized_test_accuracy" in metrics else "",
                    "personalized_test_loss": float(metrics.get("personalized_test_loss", "")) if "personalized_test_loss" in metrics else "",
                })
        
        # Write per-client test metrics to clients.csv (append columns to existing rows)
        if per_client_test_rows:
            os.makedirs("metrics", exist_ok=True)
            clients_csv = os.path.join("metrics", "clients.csv")
            
            # Read existing data if file exists
            existing_data = []
            fieldnames = [
                "round","cid","num_examples","train_loss_mean","train_accuracy_mean",
                "upload_bytes","download_bytes","comp_time_sec","y_delta_norm","c_delta_norm",
                "test_loss","test_accuracy","train_loss","train_accuracy",
                "accuracy_gap","loss_gap","personalized_test_accuracy","personalized_test_loss"
            ]
            
            if os.path.exists(clients_csv):
                with open(clients_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
                    # Update fieldnames to include any existing columns
                    if reader.fieldnames:
                        fieldnames = list(reader.fieldnames)
                        # Add missing columns
                        new_cols = ["test_loss","test_accuracy","train_loss","train_accuracy",
                                   "accuracy_gap","loss_gap","personalized_test_accuracy","personalized_test_loss"]
                        for col in new_cols:
                            if col not in fieldnames:
                                fieldnames.append(col)
            
            # Update existing data with test metrics
            for test_row in per_client_test_rows:
                # Find matching row by round and cid
                for existing_row in existing_data:
                    if (existing_row.get("round") == str(test_row["round"]) and 
                        existing_row.get("cid") == test_row["cid"]):
                        # Update all test metrics
                        for key in ["test_loss", "test_accuracy", "train_loss", "train_accuracy", 
                                   "accuracy_gap", "loss_gap", "personalized_test_accuracy", "personalized_test_loss"]:
                            if key in test_row:
                                existing_row[key] = test_row[key] if test_row[key] is not None else ""
                        break
                else:
                    # No matching row found, create new row with test metrics only
                    new_row = {field: "" for field in fieldnames}
                    new_row.update(test_row)
                    existing_data.append(new_row)
            
            # Write updated data back to file
            with open(clients_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_data)
        
        # Let parent class handle the aggregation
        return super().aggregate_evaluate(server_round, results, failures)
