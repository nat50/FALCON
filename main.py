"""Entry point: run the FALCON federated simulation with Flower.

Usage:
    python main.py

Everything is configured in falcon/config.py. For a quick smoke test keep the
defaults (small model, few clients, capped data); scale up for real experiments.
"""

import random

import flwr as fl
import numpy as np
import torch

from falcon import data
from falcon.agent import make_agent
from falcon.client import FalconClient
from falcon.config import default_config
from falcon.strategy import FalconStrategy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assign_ranks(num_clients: int, rank_pool):
    """Give each client a (possibly different) LoRA rank from the pool."""
    return {cid: rank_pool[cid % len(rank_pool)] for cid in range(num_clients)}


def load_data_and_ranks(config):
    """Prepare client datasets and per-client ranks once (reusable across runs)."""
    client_data = data.load_client_datasets(
        config.dataset_name, config.num_clients,
        config.max_train_per_client, config.max_test_per_client, config.seed,
    )
    client_ranks = assign_ranks(config.num_clients, config.client_rank_pool)
    print(f"[main] client ranks: {client_ranks}")
    return client_data, client_ranks


def run_simulation(config, client_data, client_ranks):
    """Run one Flower simulation and return its History object."""
    def client_fn(cid: str):
        client_id = int(cid)
        return FalconClient(
            client_id=client_id,
            rank=client_ranks[client_id],
            train_texts=client_data[client_id]["train"],
            test_texts=client_data[client_id]["test"],
            config=config,
        ).to_client()

    strategy = FalconStrategy(config, make_agent(config), client_ranks)
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": config.num_cpus_per_client,
            "num_gpus": config.num_gpus_per_client,
        },
    )


def main() -> None:
    config = default_config()
    set_seed(config.seed)
    client_data, client_ranks = load_data_and_ranks(config)
    run_simulation(config, client_data, client_ranks)


if __name__ == "__main__":
    main()
