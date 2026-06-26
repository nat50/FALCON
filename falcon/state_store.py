"""On-disk store for each client's PERSONAL B matrices.

B is the personalized half of LoRA and must never be uploaded unless the agent
asks for it. Flower simulation recreates client objects every round, so we persist
B on the client's local disk (keyed by client id) to keep it across rounds.
"""

import os
import pickle
from typing import Dict, Optional

import numpy as np


def _path(state_dir: str, client_id: int) -> str:
    return os.path.join(state_dir, f"client_{client_id}_B.pkl")


def save_personal_B(state_dir: str, client_id: int,
                    b_by_layer: Dict[str, np.ndarray]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    with open(_path(state_dir, client_id), "wb") as handle:
        pickle.dump(b_by_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_personal_B(state_dir: str, client_id: int
                    ) -> Optional[Dict[str, np.ndarray]]:
    path = _path(state_dir, client_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)
