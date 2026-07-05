"""Entry point: run the full FALCON federated pipeline with Flower.

Usage:
    python main.py

Everything is configured in falcon/config.py. Results and logs for each run are
written to a timestamped directory under the configured output folder.
"""

import random

import flwr as fl
import numpy as np
import torch
from flwr.common import Context

from falcon import data
from falcon.agent import make_agent
from falcon.client import FalconClient
from falcon.config import default_config
from falcon.fedit_strategy import FedITStrategy
from falcon.fedsa_strategy import FedSAStrategy
from falcon.flexlora_strategy import FlexLoRAStrategy
from falcon.results import RunLogger
from falcon.strategy import FalconStrategy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assign_ranks(num_clients: int, rank_pool):
    """Give each client a (possibly different) LoRA rank from the pool."""
    if not rank_pool:
        raise ValueError("rank_pool must not be empty")
    return {cid: rank_pool[cid % len(rank_pool)] for cid in range(num_clients)}


def assign_data_ranks(client_data, rank_pool):
    """Assign higher LoRA ranks to clients with more training examples."""
    if len(rank_pool) != len(client_data):
        raise ValueError(
            "FlexLoRA rank pool must contain exactly one rank per client: "
            f"got {len(rank_pool)} ranks for {len(client_data)} clients"
        )
    ranks_desc = sorted((int(rank) for rank in rank_pool), reverse=True)
    clients_by_data = sorted(
        client_data,
        key=lambda cid: len(client_data[cid]["train"]),
        reverse=True,
    )
    return {client_id: rank for client_id, rank in zip(clients_by_data, ranks_desc)}


def load_data_and_ranks(config):
    """Prepare client datasets and per-client ranks."""
    client_data = data.load_client_datasets(
        config.data_path, config.num_clients,
        config.eval_fraction, config.seed,
    )
    method = config.baseline_method.lower()
    if method in {"flexlora", "flexlora_data_rank"}:
        client_ranks = assign_data_ranks(client_data, config.flexlora_rank_pool)
    else:
        client_ranks = assign_ranks(config.num_clients, config.rank_pool)
    print(f"[main] client ranks: {client_ranks}")
    return client_data, client_ranks


def _round_metric_rows(history, per_client_eval):
    """Flatten Flower History into per-round metric rows."""
    losses = dict(history.losses_distributed)
    fit_metrics = history.metrics_distributed_fit
    comm = dict(fit_metrics.get("comm_cost", []))
    uploaders = dict(fit_metrics.get("num_b_uploaders", []))
    rounds = sorted(set(losses) | set(comm) | set(uploaders)
                    | set(per_client_eval))
    return [
        {
            "round": rnd,
            "mean_eval_loss": losses.get(rnd, ""),
            "comm_cost": comm.get(rnd, ""),
            "num_b_uploaders": uploaders.get(rnd, ""),
        }
        for rnd in rounds
    ]


def _per_client_rows(per_client_eval):
    rows = []
    for rnd in sorted(per_client_eval):
        for client_id, loss in sorted(per_client_eval[rnd].items()):
            rows.append({"round": rnd, "client_id": client_id, "eval_loss": loss})
    return rows


def _write_outputs(logger, config, history, strategy):
    metric_rows = _round_metric_rows(history, strategy.per_client_eval)
    logger.save_round_metrics(metric_rows)
    logger.save_per_client_eval(_per_client_rows(strategy.per_client_eval))

    losses = history.losses_distributed
    comm_series = history.metrics_distributed_fit.get("comm_cost", [])
    logger.save_summary({
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "final_eval_loss": losses[-1][1] if losses else None,
        "total_comm_cost": float(sum(value for _, value in comm_series)),
    })


def _make_strategy(config, client_ranks):
    method = config.baseline_method.lower()
    if method == "falcon":
        return FalconStrategy(config, make_agent(config), client_ranks)
    if method == "fedsa":
        return FedSAStrategy(config, client_ranks)
    if method in {"fedit", "fedit_fixed"}:
        return FedITStrategy(config, client_ranks)
    if method in {"flexlora", "flexlora_data_rank"}:
        return FlexLoRAStrategy(config, client_ranks)
    raise ValueError(f"unknown baseline_method: {config.baseline_method!r}")


def run_simulation(config, client_data, client_ranks, logger):
    """Run the configured federated simulation and persist its results."""
    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        return FalconClient(
            client_id=client_id,
            rank=client_ranks[client_id],
            train_texts=client_data[client_id]["train"],
            test_texts=client_data[client_id]["eval"],
            config=config,
        ).to_client()

    strategy = _make_strategy(config, client_ranks)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": config.num_cpus_per_client,
            "num_gpus": config.num_gpus_per_client,
        },
    )
    _write_outputs(logger, config, history, strategy)
    return history


def main() -> None:
    config = default_config()
    set_seed(config.seed)
    logger = RunLogger(config.output_dir)
    try:
        logger.dump_config(config)
        client_data, client_ranks = load_data_and_ranks(config)
        run_simulation(config, client_data, client_ranks, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
