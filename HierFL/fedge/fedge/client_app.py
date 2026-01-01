"""fedge: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedge.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition_id"]
    num_partitions = context.node_config["num_partitions"]
    # Load partitions from PARTITIONS_JSON (REQUIRED - no fallback)
    import json
    import os
    partitions_json = os.environ.get("PARTITIONS_JSON")
    if not partitions_json or not os.path.exists(partitions_json):
        raise RuntimeError(
            "PARTITIONS_JSON is required for client_app.py. "
            "Run orchestrator.py to generate user-based partitions first."
        )
    
    with open(partitions_json, 'r') as f:
        mapping = json.load(f)
    
    # For simulation, assume server_id=0 unless specified
    server_id = os.environ.get("SERVER_ID", "0")
    if str(server_id) not in mapping:
        raise RuntimeError(f"Server {server_id} not found in partitions")
    if str(partition_id) not in mapping[str(server_id)]:
        raise RuntimeError(f"Client {partition_id} not found for server {server_id}")
    
    indices = mapping[str(server_id)][str(partition_id)]
    trainloader, testloader, num_classes = load_data(
        "hhar", partition_id, num_partitions, indices=indices
    )
    local_epochs = context.run_config["local_epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
