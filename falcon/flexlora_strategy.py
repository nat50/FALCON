"""Flower strategy for the FlexLoRA baseline.

FlexLoRA keeps heterogeneous client ranks, asks every client to upload both
LoRA factors, averages the full updates dW = B @ A, then factorizes the global
update back into shared LoRA factors.
"""

import json
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from . import lora_math
from .config import Config
from .serialization import decode, encode

SELECTED_CLIENT_IDS_KEY = "selected_client_ids"
REQUEST_B_KEY = "request_b"
NEW_RANK_KEY = "new_rank"


def _partition_id(proxy: ClientProxy) -> int:
    pid = getattr(proxy, "partition_id", None)
    if pid is not None:
        return int(pid)
    return int(proxy.cid)


class FlexLoRAStrategy(fl.server.strategy.Strategy):
    def __init__(self, config: Config, client_ranks: Dict[int, int]):
        self.config = config
        self.client_ranks = client_ranks
        self.global_blob = None
        self.per_client_eval: Dict[int, Dict[int, float]] = {}

    def _server_rank(self) -> int:
        return max(int(rank) for rank in self.client_ranks.values())

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters([encode(None)])

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        global_params = ndarrays_to_parameters([encode(self.global_blob)])
        selected = sorted(int(client_id) for client_id in self.client_ranks)
        selected_json = json.dumps(selected)
        instructions = []
        for proxy in client_manager.all().values():
            client_id = _partition_id(proxy)
            fit_config = {
                REQUEST_B_KEY: 1,
                SELECTED_CLIENT_IDS_KEY: selected_json,
                NEW_RANK_KEY: self.client_ranks[client_id],
            }
            instructions.append((proxy, FitIns(global_params, fit_config)))
        print(f"[FlexLoRA] round {server_round}: requesting A+B from all clients")
        return instructions

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        payloads = [decode(parameters_to_ndarrays(res.parameters)[0])
                    for _, res in results]
        self._validate_full_upload(payloads, server_round)
        self.global_blob = self._average_full_update_blob(payloads)

        comm_cost = self._communication_cost(payloads)
        print(f"[FlexLoRA] round {server_round}: merged full updates, "
              f"comm_cost={comm_cost:.0f}")
        metrics = {
            "comm_cost": comm_cost,
            "num_b_uploaders": len(payloads),
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
        print(f"[FlexLoRA] round {server_round}: mean eval loss = {mean_loss:.4f}")
        return float(mean_loss), {"mean_eval_loss": mean_loss}

    def evaluate(self, server_round: int, parameters: Parameters
                 ) -> Optional[Tuple[float, Dict]]:
        return None

    @staticmethod
    def _has_B(payload) -> bool:
        return all(layer["B"] is not None for layer in payload["layers"].values())

    def _validate_full_upload(self, payloads, server_round: int) -> None:
        missing = sorted(
            p["client_id"] for p in payloads
            if not p["request_b"] or not self._has_B(p)
        )
        if missing:
            raise RuntimeError(
                f"round {server_round}: FlexLoRA requires A+B from every client, "
                f"but clients {missing} did not upload complete B factors"
            )

    def _layer_keys(self, payloads) -> List[str]:
        return list(payloads[0]["layers"].keys())

    def _average_full_update_blob(self, payloads):
        samples = np.asarray([p["num_examples"] for p in payloads], dtype=np.float64)
        weights = samples / (float(np.sum(samples)) or 1.0)
        blob = {}
        for key in self._layer_keys(payloads):
            delta_global = None
            first_dtype = payloads[0]["layers"][key]["A"].dtype
            for payload, weight in zip(payloads, weights):
                a_mat = payload["layers"][key]["A"]
                b_mat = payload["layers"][key]["B"]
                if b_mat.shape[1] != a_mat.shape[0]:
                    raise ValueError(
                        f"FlexLoRA layer {key} has incompatible shapes: "
                        f"B{b_mat.shape} @ A{a_mat.shape}"
                    )
                delta = b_mat @ a_mat
                if delta_global is None:
                    delta_global = np.zeros(delta.shape, dtype=np.float64)
                elif delta.shape != delta_global.shape:
                    raise ValueError(
                        f"FlexLoRA layer {key} update shape mismatch: "
                        f"expected {delta_global.shape}, got {delta.shape}"
                    )
                delta_global += weight * delta

            a_global, b_global = lora_math.factorize(delta_global, self._server_rank())
            blob[key] = (
                a_global.astype(first_dtype, copy=False),
                b_global.astype(first_dtype, copy=False),
            )
        return blob

    def _communication_cost(self, payloads) -> float:
        a_cost = sum(p["rank"] for p in payloads)
        b_cost = sum(p["rank"] for p in payloads)
        return float(a_cost + b_cost)
