"""Server-side agentic control plane: rank allocation + B-uploader selection.

Each round, the agent:
  - scores clients and allocates dynamic LoRA ranks from data scale, learning
    difficulty, and consensus novelty;
  - solves an exact 0/1 knapsack to pick which clients upload B under the
    communication budget.

A "client stat" is a dict: {"client_id": int, "align": float, "num_examples": int, "cost": float}.
"""

import math
from typing import Dict, List
import numpy as np
from .config import Config


def _minmax(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] with the epsilon from the algorithm note."""
    eps = 1e-6
    return (values - np.min(values)) / (np.max(values) - np.min(values) + eps)


class FalconAgent:
    """Agentic control plane: dynamic rank allocation + knapsack B-uploader selection."""

    def __init__(self, config: Config):
        pass

    def compute_rank_scores(
        self,
        n_samples: List[float],
        loss_before: List[float],
        loss_after: List[float],
        align_scores: List[float],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> np.ndarray:
        """Compute dynamic rank scores S_i from data, learning difficulty, and novelty."""
        n_arr = np.asarray(n_samples, dtype=np.float64)
        before = np.asarray(loss_before, dtype=np.float64)
        after = np.asarray(loss_after, dtype=np.float64)
        align = np.asarray(align_scores, dtype=np.float64)

        eps = 1e-6
        data_score = _minmax(n_arr)
        progress = (before - after) / (before + eps)
        learn_score = 1.0 - _minmax(progress)
        novelty_score = 1.0 - np.clip(align, 0.0, 1.0)

        scores = alpha * data_score + beta * learn_score + gamma * novelty_score
        return np.clip(scores, 0.0, 1.0)

    def allocate_ranks(self, rank_scores: List[float], rank_pool: List[int]) -> List[int]:
        """Map rank scores in [0, 1] to discrete ranks from the configured pool."""
        if not rank_pool:
            raise ValueError("rank_pool must not be empty")
        pool = sorted(int(rank) for rank in rank_pool)
        scores = np.clip(np.asarray(rank_scores, dtype=np.float64), 0.0, 1.0)
        indices = np.minimum((scores * len(pool)).astype(int), len(pool) - 1)
        return [pool[int(index)] for index in indices]

    def select(self, stats: List[Dict], budget: float) -> List[int]:
        """Solve an exact 0/1 knapsack: maximize total align under the budget."""
        cap = int(math.floor(budget))
        items = [(s["client_id"], max(int(s["cost"]), 1), float(s["align"])) for s in stats]

        # dp[w] = best total align achievable with total cost <= w
        dp = [0.0] * (cap + 1)
        choice = [[] for _ in range(cap + 1)]
        for cid, cost, value in items:
            for w in range(cap, cost - 1, -1):
                if dp[w - cost] + value > dp[w]:
                    dp[w] = dp[w - cost] + value
                    choice[w] = choice[w - cost] + [cid]

        best_w = max(range(cap + 1), key=lambda w: dp[w])
        chosen = choice[best_w]
        print(f"[agent] (knapsack) selected B-uploaders: {sorted(chosen)}")
        return chosen


def make_agent(config: Config):
    return FalconAgent(config)
