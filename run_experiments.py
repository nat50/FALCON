"""Run the three baselines on the SAME data and print a comparison table.

Baselines (all share one pipeline; only the B-upload policy differs):
    fedsa    - share consensus A only, keep B local (FedSA-LoRA).
    flexlora - every client uploads B every round (full aggregation).
    falcon   - the agent selects B-uploaders under a communication budget (ours).

Reported per baseline:
    final mean eval loss (lower is better)  and  total communication cost.

Usage:
    python run_experiments.py
"""

import dataclasses
import os
import shutil

from falcon.config import default_config
from main import load_data_and_ranks, run_simulation, set_seed

MODES = ["fedsa", "flexlora", "falcon"]


def _final_loss(history) -> float:
    losses = history.losses_distributed
    return losses[-1][1] if losses else float("nan")


def _total_comm(history) -> float:
    series = history.metrics_distributed_fit.get("comm_cost", [])
    return float(sum(value for _, value in series))


def main() -> None:
    base = default_config()
    set_seed(base.seed)
    client_data, client_ranks = load_data_and_ranks(base)

    results = {}
    for mode in MODES:
        print(f"\n===== running baseline: {mode} =====")
        state_dir = f"./client_state_{mode}"
        shutil.rmtree(state_dir, ignore_errors=True)  # fresh personal B per baseline
        os.makedirs(state_dir, exist_ok=True)

        config = dataclasses.replace(base, selection_mode=mode, state_dir=state_dir)
        set_seed(base.seed)  # same init for a fair comparison
        history = run_simulation(config, client_data, client_ranks)
        results[mode] = (_final_loss(history), _total_comm(history))

    print("\n===== comparison =====")
    print(f"{'mode':<10}{'final_eval_loss':>18}{'total_comm_cost':>18}")
    for mode in MODES:
        loss, comm = results[mode]
        print(f"{mode:<10}{loss:>18.4f}{comm:>18.0f}")


if __name__ == "__main__":
    main()
