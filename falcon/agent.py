"""Server-side selection agent: decide which clients upload B this round.

The agent solves an exact 0/1 knapsack: maximize total 'align' subject to the
per-round communication 'budget', where each client's cost is its rank.

A "client stat" is a dict: {"client_id": int, "align": float, "num_examples": int, "cost": float}.
"""

import math
from typing import Dict, List

from .config import Config


class KnapsackAgent:
    """Selects the B-uploader subset maximizing total align under the budget (0/1 knapsack)."""

    def __init__(self, config: Config):
        pass

    def select(self, stats: List[Dict], budget: float) -> List[int]:
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
    return KnapsackAgent(config)
