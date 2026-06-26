"""Custom Flower strategy implementing the FALCON server logic.

Responsibilities:
  - configure_fit: broadcast the global blob and tell each client whether to send B.
  - aggregate_fit: build the consensus subspace from ALL A's, merge the B-clients'
    content, factorize into the new global (A_global, B_global), then run the agent
    to choose who uploads B next round.

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
from .client import REQUEST_B_KEY, SELECTED_CLIENT_IDS_KEY
from .config import Config
from .serialization import decode, encode


class FalconStrategy(fl.server.strategy.Strategy):
    def __init__(self, config: Config, agent, client_ranks: Dict[int, int]):
        self.config = config
        self.agent = agent
        self.client_ranks = client_ranks
        self.total_rank = float(sum(client_ranks.values()))
        self.budget = max(
            config.b_budget_fraction * self.total_rank,
            float(min(client_ranks.values())),
        )
        self.mode = config.selection_mode
        self.global_blob = None
        self.selected = self._initial_selection()
        self.requested_by_round: Dict[int, List[int]] = {}

    # ----- selection bookkeeping -----
    def _all_ids(self) -> List[int]:
        return list(self.client_ranks.keys())

    def _initial_selection(self) -> List[int]:
        """Pick the round-1 B-uploaders according to the baseline mode."""
        if self.mode == "flexlora":
            return self._all_ids()
        if self.mode == "fedsa":
            return []
        return self._bootstrap_selection()  # falcon

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
        """Choose next round's B-uploaders according to the baseline mode."""
        if self.mode == "flexlora":
            return [s["client_id"] for s in stats]
        if self.mode == "fedsa":
            return []
        chosen = self.agent.select(stats, self.budget)  # falcon
        if not chosen:
            chosen = [max(stats, key=lambda s: s["align"])["client_id"]]
        return chosen

    # ----- Flower API -----
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters([encode(None)])

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        global_params = ndarrays_to_parameters([encode(self.global_blob)])
        selected = sorted(int(client_id) for client_id in self.selected)
        selected_set = set(selected)
        selected_json = json.dumps(selected)
        self.requested_by_round[server_round] = selected
        instructions = []
        for proxy in client_manager.all().values():
            request_b = int(proxy.cid) in selected_set
            fit_config = {
                REQUEST_B_KEY: 1 if request_b else 0,
                SELECTED_CLIENT_IDS_KEY: selected_json,
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
        b_client_ids = [
            p["client_id"] for p in payloads
            if p["client_id"] in requested_ids and p["request_b"] and self._has_B(p)
        ]
        if b_client_ids:
            self.global_blob = self._merge_all_layers(
                payloads, align_by_client, b_client_ids)
        else:
            # No B this round (e.g. FedSA mode): share the consensus A only.
            self.global_blob = self._consensus_only_blob(payloads)

        stats = [{
            "client_id": p["client_id"],
            "align": align_by_client[p["client_id"]],
            "num_examples": p["num_examples"],
            "cost": float(p["rank"]),
        } for p in payloads]
        self.selected = self._decide_selection(stats)

        comm_cost = self._communication_cost(payloads, b_client_ids)
        print(f"[server] round {server_round}: merged "
              f"(B from {sorted(b_client_ids)}), comm_cost={comm_cost:.0f}")
        metrics = {"comm_cost": comm_cost, "num_b_uploaders": len(b_client_ids)}
        return ndarrays_to_parameters([encode(self.global_blob)]), metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        global_params = ndarrays_to_parameters([encode(self.global_blob)])
        eval_ins = EvaluateIns(global_params, {})
        return [(proxy, eval_ins) for proxy in client_manager.all().values()]

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}
        total = sum(res.num_examples for _, res in results)
        weighted = sum(res.loss * res.num_examples for _, res in results)
        mean_loss = weighted / max(total, 1)
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
            v_basis = lora_math.consensus_subspace(a_all, self.config.global_rank)
            proj = lora_math.projector(v_basis)
            for p in payloads:
                align_sum[p["client_id"]] += lora_math.alignment_score(
                    p["layers"][key]["A"], proj)
        return {cid: total / max(len(keys), 1) for cid, total in align_sum.items()}

    def _merge_all_layers(self, payloads, align_by_client, b_client_ids):
        keys = self._layer_keys(payloads)
        id_to_payload = {p["client_id"]: p for p in payloads}
        weights = [align_by_client[cid] * id_to_payload[cid]["num_examples"]
                   for cid in b_client_ids]

        blob = {}
        for key in keys:
            a_all = [p["layers"][key]["A"] for p in payloads]
            b_list = [id_to_payload[cid]["layers"][key]["B"] for cid in b_client_ids]
            a_list = [id_to_payload[cid]["layers"][key]["A"] for cid in b_client_ids]
            blob[key] = lora_math.merge_layer(
                a_all, b_list, a_list, weights, self.config.global_rank)
        return blob

    def _consensus_only_blob(self, payloads):
        """FedSA-style global: shared consensus A per layer, no shared B (B stays local)."""
        blob = {}
        for key in self._layer_keys(payloads):
            a_all = [p["layers"][key]["A"] for p in payloads]
            a_global = lora_math.consensus_subspace(a_all, self.config.global_rank)
            blob[key] = (a_global, None)
        return blob

    def _communication_cost(self, payloads, b_client_ids) -> float:
        a_cost = sum(p["rank"] for p in payloads)  # everyone uploads A
        b_cost = sum(p["rank"] for p in payloads if p["client_id"] in b_client_ids)
        return float(a_cost + b_cost)
