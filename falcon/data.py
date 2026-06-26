"""Non-IID client data built from databricks-dolly-15k.

Each Dolly `category` (open_qa, classification, summarization, ...) becomes one
client. This yields a natural non-IID split: instruction-following is the shared
"common knowledge", while the per-category style/format is the "private" part.

Each client returns plain lists of formatted text strings, so the client trainer
stays simple and framework-agnostic.
"""

import random
from typing import Dict, List

from datasets import load_dataset

PROMPT_WITH_CONTEXT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{context}\n\n"
    "### Response:\n{response}"
)
PROMPT_NO_CONTEXT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)


def format_example(example: dict) -> str:
    """Render one Dolly row into a single training string."""
    context = (example.get("context") or "").strip()
    if context:
        return PROMPT_WITH_CONTEXT.format(
            instruction=example["instruction"].strip(),
            context=context,
            response=example["response"].strip(),
        )
    return PROMPT_NO_CONTEXT.format(
        instruction=example["instruction"].strip(),
        response=example["response"].strip(),
    )


def load_client_datasets(
    dataset_name: str,
    num_clients: int,
    max_train: int,
    max_test: int,
    seed: int,
) -> Dict[int, Dict[str, List[str]]]:
    """Group Dolly by category and turn the top-N categories into clients.

    Returns:
        {client_id: {"train": [str, ...], "test": [str, ...], "category": str}}
    """
    rng = random.Random(seed)
    dataset = load_dataset(dataset_name, split="train")

    by_category: Dict[str, List[str]] = {}
    for row in dataset:
        by_category.setdefault(row["category"], []).append(format_example(row))

    # Pick the largest categories so every client has enough data.
    categories = sorted(by_category, key=lambda c: len(by_category[c]), reverse=True)
    categories = categories[:num_clients]

    clients: Dict[int, Dict[str, List[str]]] = {}
    for client_id, category in enumerate(categories):
        texts = by_category[category]
        rng.shuffle(texts)
        test = texts[:max_test]
        train = texts[max_test : max_test + max_train]
        clients[client_id] = {"train": train, "test": test, "category": category}
        print(
            f"[data] client {client_id} <- category '{category}': "
            f"{len(train)} train / {len(test)} test"
        )
    return clients


def build_global_testset(
    clients: Dict[int, Dict[str, List[str]]], per_client: int, seed: int
) -> List[str]:
    """Mix a few test samples from every client to measure shared knowledge."""
    rng = random.Random(seed)
    mixed: List[str] = []
    for client in clients.values():
        mixed.extend(client["test"][:per_client])
    rng.shuffle(mixed)
    return mixed
