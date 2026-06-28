"""Federated client datasets built from Fed-WildChat (FedLLM-Bench).

The raw file is a JSON object mapping each source user id to a list of
{"instruction", "response"} samples. Users are merged into a fixed number of
clients (balanced by sample count) to form the federation, and each client
holds out a fraction of its data for evaluation.

Each client returns plain lists of formatted text strings, so the client trainer
stays simple and framework-agnostic.
"""

import json
import random
from typing import Dict, List, Tuple

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)


def format_example(example: dict) -> str:
    """Render one sample into a single training string."""
    return PROMPT_TEMPLATE.format(
        instruction=str(example.get("instruction", "")).strip(),
        response=str(example.get("response", "")).strip(),
    )


def _load_users(data_path: str) -> Dict[str, List[str]]:
    """Read the raw JSON and return {user_id: [formatted_text, ...]}."""
    with open(data_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        user_id: [format_example(sample) for sample in samples]
        for user_id, samples in raw.items()
    }


def _merge_users_into_clients(
    users: Dict[str, List[str]], num_clients: int
) -> Tuple[List[List[str]], List[int]]:
    """Greedily assign whole users to clients, balancing total sample counts."""
    texts_bins: List[List[str]] = [[] for _ in range(num_clients)]
    user_counts = [0] * num_clients
    sizes = [0] * num_clients
    ordered = sorted(users.values(), key=len, reverse=True)
    for texts in ordered:
        target = min(range(num_clients), key=lambda i: sizes[i])
        texts_bins[target].extend(texts)
        sizes[target] += len(texts)
        user_counts[target] += 1
    return texts_bins, user_counts


def load_client_datasets(
    data_path: str,
    num_clients: int,
    eval_fraction: float,
    seed: int,
) -> Dict[int, Dict[str, List[str]]]:
    """Build per-client train/eval splits from the Fed-WildChat JSON file.

    Returns:
        {client_id: {"train": [str, ...], "eval": [str, ...]}}
    """
    rng = random.Random(seed)
    users = _load_users(data_path)
    texts_bins, user_counts = _merge_users_into_clients(users, num_clients)

    clients: Dict[int, Dict[str, List[str]]] = {}
    for client_id, texts in enumerate(texts_bins):
        rng.shuffle(texts)
        n_eval = int(len(texts) * eval_fraction)
        eval_texts = texts[:n_eval]
        train_texts = texts[n_eval:]
        clients[client_id] = {"train": train_texts, "eval": eval_texts}
        print(
            f"[data] client {client_id}: {len(train_texts)} train / "
            f"{len(eval_texts)} eval ({user_counts[client_id]} users)"
        )
    return clients
