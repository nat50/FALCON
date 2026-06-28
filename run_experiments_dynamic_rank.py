"""Compare FALCON with and without Dynamic Rank Allocation.

This keeps the same data, seed, and simulation pipeline as run_experiments.py.
The only intended algorithmic difference is:
    falcon_static       -> fixed rank, legacy weight align_i * n_i
    falcon_dynamic_rank -> dynamic rank, new weight S_i * n_i

Usage:
    python run_experiments_dynamic_rank.py
"""

import dataclasses
import os
import shutil

from falcon.config import default_config
from main import load_data_and_ranks, run_simulation, set_seed


EXPERIMENTS = [
    (
        "falcon_static",
        {
            "selection_mode": "falcon",
            "use_dynamic_rank": False,
        },
    ),
    (
        "falcon_dynamic_rank",
        {
            "selection_mode": "falcon",
            "use_dynamic_rank": True,
            "rank_alpha": 0.3,
            "rank_beta": 0.3,
            "rank_gamma": 0.4,
            "rank_pool": [4, 8, 16, 32],
            "global_rank": 32,
        },
    ),
]


def _final_loss(history) -> float:
    losses = history.losses_distributed
    return losses[-1][1] if losses else float("nan")


def _total_comm(history) -> float:
    series = history.metrics_distributed_fit.get("comm_cost", [])
    return float(sum(value for _, value in series))


def _final_mean_rank_score(history) -> float:
    series = history.metrics_distributed_fit.get("mean_rank_score", [])
    return series[-1][1] if series else float("nan")


def main() -> None:
    base = default_config()
    set_seed(base.seed)
    client_data, client_ranks = load_data_and_ranks(base)

    results = {}
    for name, overrides in EXPERIMENTS:
        print(f"\n===== running experiment: {name} =====")
        state_dir = f"./client_state_{name}"
        shutil.rmtree(state_dir, ignore_errors=True)
        os.makedirs(state_dir, exist_ok=True)

        config = dataclasses.replace(base, **overrides, state_dir=state_dir)
        set_seed(base.seed)
        history = run_simulation(config, client_data, client_ranks.copy())
        results[name] = (
            _final_loss(history),
            _total_comm(history),
            _final_mean_rank_score(history),
        )

    print("\n===== dynamic rank comparison =====")
    print(
        f"{'experiment':<22}"
        f"{'final_eval_loss':>18}"
        f"{'total_comm_cost':>18}"
        f"{'final_rank_score':>18}"
    )
    for name, _overrides in EXPERIMENTS:
        loss, comm, rank_score = results[name]
        print(f"{name:<22}{loss:>18.4f}{comm:>18.0f}{rank_score:>18.4f}")


if __name__ == "__main__":
    main()
