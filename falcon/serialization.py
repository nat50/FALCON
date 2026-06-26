"""Serialize arbitrary client/server payloads as a single uint8 NumPy array.

Client adapters have heterogeneous shapes (different ranks), so instead of fighting
Flower's array-aligned aggregation we pack each payload (a plain dict) into one
opaque byte tensor. The custom strategy unpacks and merges it by hand.
"""

import pickle
from typing import Any

import numpy as np


def encode(obj: Any) -> np.ndarray:
    """Pickle a Python object into a 1-D uint8 array."""
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(raw, dtype=np.uint8).copy()


def decode(arr: np.ndarray) -> Any:
    """Inverse of `encode`. Returns the original Python object."""
    if arr is None or arr.size == 0:
        return None
    return pickle.loads(arr.astype(np.uint8).tobytes())
