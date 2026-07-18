"""Standalone sanity check for the LoRA merge math (no ML deps, just NumPy).

Run:
    python scripts/check_merge.py

It builds fake clients with DIFFERENT ranks, runs the full merge, truncates the
global factors back to each client's rank, and prints shapes plus reconstruction
error. Use it to verify the core algorithm before launching the full simulation.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falcon import lora_math  # noqa: E402

D_OUT, K_IN, GLOBAL_RANK = 12, 8, 6
CLIENT_RANKS = [2, 4, 6]


def make_client(rank: int, rng: np.random.Generator):
    a = rng.normal(size=(rank, K_IN))
    b = rng.normal(size=(D_OUT, rank))
    return a, b


def main() -> None:
    rng = np.random.default_rng(0)
    clients = [make_client(r, rng) for r in CLIENT_RANKS]
    a_all = [a for a, _ in clients]

    # Pretend only the last two clients uploaded B.
    b_clients = [b for _, b in clients[1:]]
    a_clients = [a for a, _ in clients[1:]]
    n_samples = [10, 20]

    a_global, b_global = lora_math.merge_layer(
        a_all, b_clients, a_clients, n_samples, GLOBAL_RANK)
    print(f"A_global shape = {a_global.shape} (expected ({GLOBAL_RANK}, {K_IN}))")
    print(f"B_global shape = {b_global.shape} (expected ({D_OUT}, {GLOBAL_RANK}))")

    delta_global = b_global @ a_global
    print(f"||B_global @ A_global||_F = {np.linalg.norm(delta_global):.4f}")

    for rank in CLIENT_RANKS:
        a_t, b_t = lora_math.truncate_factors(a_global, b_global, rank)
        print(f"truncate to rank {rank}: A={a_t.shape}, B={b_t.shape}")

    # Alignment: a vector inside the consensus subspace should score ~1.
    v_basis = lora_math.consensus_subspace(a_all, GLOBAL_RANK)
    proj = lora_math.projector(v_basis)
    inside = lora_math.alignment_score(a_all[0], proj)
    outside = lora_math.alignment_score(rng.normal(size=(2, K_IN)), proj)
    print(f"alignment (client A, inside) = {inside:.3f}")
    print(f"alignment (random vector)    = {outside:.3f}")
    print("OK: merge math runs end-to-end.")


if __name__ == "__main__":
    main()
