"""Custom Flower strategy implementing the FALCON server logic.

Responsibilities:
  - configure_fit: broadcast the global blob and tell each client whether to send B.
  - aggregate_fit: build the consensus subspace from ALL A's, merge the B-clients'
    content, factorize into the new global (A_global, B_global), then run the agent
    to allocate next round's ranks and choose who uploads B next round.
  - aggregate_evaluate: average per-client eval loss and record it per round.

Communication cost is modelled per layer as proportional to rank (d is constant
across clients), so we use rank directly as the cost unit.
"""

import json
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from . import lora_math
from .client import NEW_RANK_KEY, REQUEST_B_KEY, SELECTED_CLIENT_IDS_KEY
from .config import Config
from .serialization import decode, encode


def _partition_id(proxy: ClientProxy) -> int:
    """Return the logical client index (0..num_clients-1).

    Flower Ray VCE sets proxy.cid to a node_id hash; partition_id holds the
    dataset index that client_ranks and payloads use.
    """
    pid = getattr(proxy, "partition_id", None)
    if pid is not None:
        return int(pid)
    return int(proxy.cid)


class FalconStrategy(fl.server.strategy.Strategy):
    def __init__(self, config: Config, agent, client_ranks: Dict[int, int]):
        self.config = config
        self.agent = agent
        self.client_ranks = client_ranks
        self.total_rank = 0.0
        self.budget = 0.0
        self._refresh_budget()
        self.global_blob = None
        self.selected = self._bootstrap_selection()
        self.requested_by_round: Dict[int, List[int]] = {}
        self.per_client_eval: Dict[int, Dict[int, float]] = {}
        self.pending_ranks: Dict[int, int] = {}

    # ----- selection bookkeeping -----
    def _server_rank(self) -> int:
        return max(self.config.rank_pool)

    def _apply_pending_ranks(self) -> None:
        """Promote ranks computed last round so fit/eval use the same rank."""
        if not self.pending_ranks:
            return
        self.client_ranks.update(self.pending_ranks)
        self.pending_ranks.clear()
        self._refresh_budget()

    def _refresh_budget(self, ranks: Optional[Dict[int, int]] = None) -> None:
        active = ranks if ranks is not None else self.client_ranks
        self.total_rank = float(sum(active.values()))
        self.budget = max(
            self.config.b_budget_fraction * self.total_rank,
            float(min(active.values())),
        )

    def _bootstrap_selection(self) -> List[int]:
        """Round 1 has no alignment yet: spend the budget on the cheapest clients."""
        by_cost = sorted(self.client_ranks.items(), key=lambda kv: kv[1])
        chosen, spent = [], 0.0
        for client_id, rank in by_cost:
            if spent + rank <= self.budget:
                chosen.append(client_id)
                spent += rank
        return chosen or [by_cost[0][0]]

    def _decide_selection(self, stats) -> List[int]:
        """Ask the agent to choose next round's B-uploaders under the budget."""
        chosen = self.agent.select(stats, self.budget)
        if not chosen:
            chosen = [max(stats, key=lambda s: s["align"])["client_id"]]
        return chosen

    # ----- Flower API -----
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters([encode(None)])

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        self._apply_pending_ranks()
        global_params = ndarrays_to_parameters([encode(self.global_blob)])
        selected = sorted(int(client_id) for client_id in self.selected)
        selected_set = set(selected)
        selected_json = json.dumps(selected)
        self.requested_by_round[server_round] = selected
        instructions = []
        for proxy in client_manager.all().values():
            client_id = _partition_id(proxy)
            request_b = client_id in selected_set
            fit_config = {
                REQUEST_B_KEY: 1 if request_b else 0,
                SELECTED_CLIENT_IDS_KEY: selected_json,
                NEW_RANK_KEY: self.client_ranks[client_id],
            }
            fit_ins = FitIns(global_params, fit_config)
            instructions.append((proxy, fit_ins))
        print(f"[server] round {server_round}: requesting B from {selected}")
        return instructions

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        payloads = [decode(parameters_to_ndarrays(res.parameters)[0])
                    for _, res in results]

        requested_ids = set(self.requested_by_round.get(server_round, self.selected))
        self._validate_b_uploads(payloads, requested_ids, server_round)
        align_by_client = self._compute_alignment(payloads)
        rank_scores_by_client = self._compute_rank_scores(payloads, align_by_client)
        new_ranks = self._allocate_new_ranks(payloads, rank_scores_by_client)
        b_client_ids = [
            p["client_id"] for p in payloads
            if p["client_id"] in requested_ids and p["request_b"] and self._has_B(p)
        ]
        if b_client_ids:
            self.global_blob = self._merge_all_layers(payloads, b_client_ids)
        else:
            # No B available this round: share the consensus A only.
            self.global_blob = self._consensus_only_blob(payloads)

        self.pending_ranks = new_ranks
        next_ranks = {**self.client_ranks, **new_ranks}
        self._refresh_budget(next_ranks)
        stats = [{
            "client_id": p["client_id"],
            "align": align_by_client[p["client_id"]],
            "num_examples": p["num_examples"],
            "rank": next_ranks[p["client_id"]],
            "cost": float(next_ranks[p["client_id"]]),
        } for p in payloads]
        self.selected = self._decide_selection(stats)

        comm_cost = self._communication_cost(payloads, b_client_ids)
        print(f"[server] round {server_round}: merged "
              f"(B from {sorted(b_client_ids)}), comm_cost={comm_cost:.0f}")
        metrics = {
            "comm_cost": comm_cost,
            "num_b_uploaders": len(b_client_ids),
        }
        return ndarrays_to_parameters([encode(self.global_blob)]), metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        global_params = ndarrays_to_parameters([encode(self.global_blob)])
        instructions = []
        for proxy in client_manager.all().values():
            client_id = _partition_id(proxy)
            eval_config = {NEW_RANK_KEY: self.client_ranks[client_id]}
            instructions.append((proxy, EvaluateIns(global_params, eval_config)))
        return instructions

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}
        total = sum(res.num_examples for _, res in results)
        weighted = sum(res.loss * res.num_examples for _, res in results)
        mean_loss = weighted / max(total, 1)
        self.per_client_eval[server_round] = {
            int(res.metrics["client_id"]): float(res.loss)
            for _, res in results
        }
        print(f"[server] round {server_round}: mean eval loss = {mean_loss:.4f}")
        return float(mean_loss), {"mean_eval_loss": mean_loss}

    def evaluate(self, server_round: int, parameters: Parameters
                 ) -> Optional[Tuple[float, Dict]]:
        return None  # evaluation is done client-side

    # ----- merge helpers -----
    @staticmethod
    def _has_B(payload) -> bool:
        return all(layer["B"] is not None for layer in payload["layers"].values())

    def _validate_b_uploads(self, payloads, requested_ids, server_round: int) -> None:
        payload_by_id = {p["client_id"]: p for p in payloads}
        received_ids = set(payload_by_id)
        missing_payloads = sorted(client_id for client_id in requested_ids
                                  if client_id not in received_ids)
        if missing_payloads:
            raise RuntimeError(
                f"round {server_round}: server requested B from "
                f"{missing_payloads}, but those clients did not return payloads"
            )

        missing_selected = sorted(
            client_id for client_id in requested_ids
            if client_id in received_ids
            and not self._has_B(payload_by_id[client_id])
        )
        if missing_selected:
            raise RuntimeError(
                f"round {server_round}: server requested B from "
                f"{missing_selected}, but their payloads did not include B"
            )

        unexpected = sorted(
            p["client_id"] for p in payloads
            if p["client_id"] not in requested_ids and self._has_B(p)
        )
        if unexpected:
            raise RuntimeError(
                f"round {server_round}: clients {unexpected} uploaded B "
                "without being selected"
            )

    def _layer_keys(self, payloads) -> List[str]:
        return list(payloads[0]["layers"].keys())

    def _compute_alignment(self, payloads) -> Dict[int, float]:
        """Average alignment-with-consensus across all layers, per client."""
        keys = self._layer_keys(payloads)
        align_sum = {p["client_id"]: 0.0 for p in payloads}
        for key in keys:
            a_all = [p["layers"][key]["A"] for p in payloads]
            v_basis = lora_math.consensus_subspace(a_all, self._server_rank())
            proj = lora_math.projector(v_basis)
            for p in payloads:
                align_sum[p["client_id"]] += lora_math.alignment_score(
                    p["layers"][key]["A"], proj)
        return {cid: total / max(len(keys), 1) for cid, total in align_sum.items()}

    def _compute_rank_scores(self, payloads, align_by_client) -> Dict[int, float]:
        scores = self.agent.compute_rank_scores(
            [p["num_examples"] for p in payloads],
            [p["loss_before"] for p in payloads],
            [p["loss_after"] for p in payloads],
            [align_by_client[p["client_id"]] for p in payloads],
            self.config.rank_alpha,
            self.config.rank_beta,
            self.config.rank_gamma,
        )
        return {p["client_id"]: float(score) for p, score in zip(payloads, scores)}

    def _allocate_new_ranks(self, payloads, rank_scores_by_client) -> Dict[int, int]:
        ranks = self.agent.allocate_ranks(
            [rank_scores_by_client[p["client_id"]] for p in payloads],
            self.config.rank_pool,
        )
        return {p["client_id"]: rank for p, rank in zip(payloads, ranks)}

    def _merge_all_layers(self, payloads, b_client_ids):
        keys = self._layer_keys(payloads)
        id_to_payload = {p["client_id"]: p for p in payloads}
        n_samples = [id_to_payload[cid]["num_examples"] for cid in b_client_ids]

        blob = {}
        for key in keys:
            a_all = [p["layers"][key]["A"] for p in payloads]
            b_list = [id_to_payload[cid]["layers"][key]["B"] for cid in b_client_ids]
            a_list = [id_to_payload[cid]["layers"][key]["A"] for cid in b_client_ids]
            blob[key] = lora_math.merge_layer(
                a_all, b_list, a_list, n_samples, self._server_rank(),
            )
        return blob

    def _consensus_only_blob(self, payloads):
        """Fallback global with shared consensus A per layer and no shared B."""
        blob = {}
        for key in self._layer_keys(payloads):
            a_all = [p["layers"][key]["A"] for p in payloads]
            a_global = lora_math.consensus_subspace(a_all, self._server_rank())
            blob[key] = (a_global, None)
        return blob

    def _communication_cost(self, payloads, b_client_ids) -> float:
        a_cost = sum(p["rank"] for p in payloads)  # everyone uploads A
        b_cost = sum(p["rank"] for p in payloads if p["client_id"] in b_client_ids)
        return float(a_cost + b_cost)
