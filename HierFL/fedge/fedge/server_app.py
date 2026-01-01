"""fedge: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedge.task import Net, get_weights
import pickle

class LeafFedAvg(FedAvg):
    def __init__(self, server_id, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.server_id = server_id
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if rnd == self.num_rounds:
            nds = parameters_to_ndarrays(aggregated)
            with open(f"server_{self.server_id}.pkl", "wb") as f:
                pickle.dump(nds, f)
        return aggregated

def weighted_average(metrics:List[Tuple[int,Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum([num_examples for num_examples, _ in metrics])
    return {"accuracy": sum(accuracies) / total_examples}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    # How many clients this leaf-server expects
    clients_per_server = context.node_config["clients_per_server"]

    # Initialize HHAR model parameters (6 input channels, 6 classes)
    ndarrays = get_weights(Net(in_ch=6, num_classes=6))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = LeafFedAvg(
        server_id=context.node_config["server_id"],
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=clients_per_server,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
