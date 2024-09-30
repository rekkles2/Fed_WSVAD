import argparse
from typing import List, Tuple
from model import model_generater
import flwr as fl
from flwr.common import Metrics
import torch
import flwr as fl
from flwr.common import ndarray_to_bytes, Parameters

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.0.102:8080",
    help=f"gRPC server address (deafault '192.168.0.101:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=6,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 4)",
)


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 10,  # Number of local epochs done by clients
        "batch_size": 1,  # Batch size to use by clients during fit()
    }
    return config

def model_to_flower_params(model_state_dict):
    tensors = [ndarray_to_bytes(v.numpy()) for v in model_state_dict.values()]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def main():
    args = parser.parse_args()
    # Gets a model state dictionary
    # Define strategy with the converted parameters
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
        grpc_max_message_length=736870912,
    )


if __name__ == "__main__":
    main()






