"""Flower client: adopt the shared A, keep a personal B, train locally, upload.

Per round:
  1. Receive the global (A_global, B_global) blob from the server.
  2. Set the shared A (truncated to this client's rank). Load the personal B from
     disk, or initialize it from the truncated global B on the first round.
  3. Fine-tune locally on private data.
  4. Save the personal B back to disk (it stays local).
  5. Upload A always; upload B only if the agent requested it this round.
"""

import json
from typing import Dict, List

import flwr as fl
import numpy as np

from . import lora_math, modeling, state_store
from .config import Config
from .serialization import decode, encode

SELECTED_CLIENT_IDS_KEY = "selected_client_ids"
REQUEST_B_KEY = "request_b"
NEW_RANK_KEY = "new_rank"


class FalconClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, rank: int, train_texts: List[str],
                 test_texts: List[str], config: Config):
        self.client_id = client_id
        self.rank = rank
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.config = config
        self.model, self.tokenizer = modeling.build_model(
            config.client_model_name, rank, config.lora_target_modules,
            rank, config.lora_dropout, config.device,
        )
        if config.freeze_shared_A:
            modeling.freeze_A(self.model)

    def _ensure_rank(self, rank: int) -> None:
        """Rebuild the local LoRA adapter if the server assigned a new rank."""
        if rank == self.rank:
            return
        self.rank = rank
        self.model, self.tokenizer = modeling.build_model(
            self.config.client_model_name, rank, self.config.lora_target_modules,
            rank, self.config.lora_dropout, self.config.device,
        )
        if self.config.freeze_shared_A:
            modeling.freeze_A(self.model)

    def _rank_from_config(self, config: Dict) -> int:
        """Read the server-assigned rank, defaulting to the current local rank."""
        raw = config.get(NEW_RANK_KEY, self.rank)
        try:
            rank = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {NEW_RANK_KEY}: {raw!r}") from exc
        if rank <= 0:
            raise ValueError(f"{NEW_RANK_KEY} must be positive, got {rank}")
        return rank

    def _apply_global(self, global_blob) -> None:
        """Install the shared A and pick B (personal if available, else shared init)."""
        if global_blob is None:
            return  # cold start: keep the random LoRA init
        personal_b = state_store.load_personal_B(self.config.state_dir, self.client_id)
        for key, (a_global, b_global) in global_blob.items():
            if a_global.shape[0] < self.rank:
                raise ValueError(
                    f"global A for {key} has rank {a_global.shape[0]}, "
                    f"cannot serve client rank {self.rank}"
                )
            modeling.set_lora_factor(self.model, key, "A", a_global[: self.rank, :])

            if personal_b and key in personal_b:
                b_saved = personal_b[key]
                if b_saved.shape[1] >= self.rank:
                    b_local = b_saved[:, : self.rank]   # keep/truncate my personalized B
                elif b_global is not None and b_global.shape[1] >= self.rank:
                    b_local = b_global[:, : self.rank].copy()
                    b_local[:, : b_saved.shape[1]] = b_saved
                else:
                    continue
            elif b_global is not None:
                if b_global.shape[1] < self.rank:
                    raise ValueError(
                        f"global B for {key} has rank {b_global.shape[1]}, "
                        f"cannot serve client rank {self.rank}"
                    )
                b_local = b_global[:, : self.rank]      # warm-start from shared B
            else:
                continue                                 # FedSA + no personal B yet: keep current
            modeling.set_lora_factor(self.model, key, "B", b_local)

    def _save_personal_B(self, factors: Dict[str, tuple]) -> None:
        b_by_layer = {key: b for key, (_, b) in factors.items()}
        state_store.save_personal_B(self.config.state_dir, self.client_id, b_by_layer)

    def _should_upload_B(self, config: Dict) -> bool:
        """Return whether this client was selected to upload B this round."""
        if SELECTED_CLIENT_IDS_KEY in config:
            try:
                selected = json.loads(str(config[SELECTED_CLIENT_IDS_KEY]))
                selected_ids = {int(client_id) for client_id in selected}
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid {SELECTED_CLIENT_IDS_KEY}: "
                    f"{config[SELECTED_CLIENT_IDS_KEY]!r}"
                ) from exc
            return self.client_id in selected_ids

        if REQUEST_B_KEY not in config:
            raise KeyError(
                f"missing fit config '{REQUEST_B_KEY}' or "
                f"'{SELECTED_CLIENT_IDS_KEY}'"
            )

        raw = config[REQUEST_B_KEY]
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return int(raw) == 1
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes"}
        raise ValueError(f"invalid {REQUEST_B_KEY}: {raw!r}")

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self._ensure_rank(self._rank_from_config(config))
        self._apply_global(decode(parameters[0]))
        loss_before = modeling.eval_loss(
            self.model, self.tokenizer, self.train_texts,
            self.config.local_batch_size, self.config.max_seq_len, self.config.device,
        )
        train_loss = modeling.train_local(
            self.model, self.tokenizer, self.train_texts,
            self.config.local_lr, self.config.local_epochs,
            self.config.local_batch_size, self.config.max_seq_len, self.config.device,
        )
        loss_after = modeling.eval_loss(
            self.model, self.tokenizer, self.train_texts,
            self.config.local_batch_size, self.config.max_seq_len, self.config.device,
        )

        factors = modeling.get_lora_AB(self.model)
        self._save_personal_B(factors)

        request_b = self._should_upload_B(config)
        layers = {}
        for key, (a_mat, b_mat) in factors.items():
            layers[key] = {"A": a_mat, "B": b_mat if request_b else None}

        payload = {
            "client_id": self.client_id,
            "rank": self.rank,
            "num_examples": len(self.train_texts),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "request_b": request_b,
            "layers": layers,
        }
        metrics = {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "loss_before": loss_before,
            "loss_after": loss_after,
        }
        print(f"[client {self.client_id}] trained "
              f"(loss_before={loss_before:.4f}, loss_after={loss_after:.4f}, "
              f"sent_B={request_b}, rank={self.rank})")
        return [encode(payload)], len(self.train_texts), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        self._apply_global(decode(parameters[0]))
        loss = modeling.eval_loss(
            self.model, self.tokenizer, self.test_texts,
            self.config.local_batch_size, self.config.max_seq_len, self.config.device,
        )
        print(f"[client {self.client_id}] eval loss={loss:.4f}")
        return float(loss), len(self.test_texts), {
            "client_id": self.client_id, "eval_loss": loss,
        }
