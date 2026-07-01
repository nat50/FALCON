"""Federated client datasets built from Databricks Dolly.

Dolly samples include a category label. Whole categories are assigned to a
fixed number of clients, balancing sample counts while keeping every category
inside exactly one client. Each client holds out a fraction of its data for
evaluation.

Each client returns plain lists of formatted text strings, so the client trainer
stays simple and framework-agnostic.
"""

from collections import defaultdict
import random
from typing import Dict, List, Tuple

from datasets import load_dataset  # type: ignore[reportMissingImports]

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "{context_block}"
    "### Response:\n{response}"
)


def format_example(example: dict) -> str:
    """Render one sample into a single training string."""
    context = str(example.get("context", "")).strip()
    context_block = f"### Context:\n{context}\n\n" if context else ""
    return PROMPT_TEMPLATE.format(
        instruction=str(example.get("instruction", "")).strip(),
        context_block=context_block,
        response=str(example.get("response", "")).strip(),
    )


def _load_categories(data_path: str) -> Dict[str, List[str]]:
    """Load Dolly and return {category: [formatted_text, ...]}."""
    dataset = load_dataset(data_path, split="train")
    categories: Dict[str, List[str]] = defaultdict(list)
    for sample in dataset:
        category = str(sample.get("category", "uncategorized")).strip()
        categories[category or "uncategorized"].append(format_example(sample))
    return dict(categories)


def _merge_categories_into_clients(
    categories: Dict[str, List[str]], num_clients: int
) -> Tuple[List[List[str]], List[List[str]]]:
    """Greedily assign whole categories to clients by total sample counts."""
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")

    texts_bins: List[List[str]] = [[] for _ in range(num_clients)]
    category_bins: List[List[str]] = [[] for _ in range(num_clients)]
    sizes = [0] * num_clients
    ordered = sorted(categories.items(), key=lambda item: len(item[1]), reverse=True)
    for category, texts in ordered:
        target = min(range(num_clients), key=lambda i: sizes[i])
        texts_bins[target].extend(texts)
        category_bins[target].append(category)
        sizes[target] += len(texts)
    return texts_bins, category_bins


def load_client_datasets(
    data_path: str,
    num_clients: int,
    eval_fraction: float,
    seed: int,
) -> Dict[int, Dict[str, List[str]]]:
    """Build per-client train/eval splits from Dolly categories.

    Returns:
        {client_id: {"train": [str, ...], "eval": [str, ...]}}
    """
    rng = random.Random(seed)
    categories = _load_categories(data_path)
    texts_bins, category_bins = _merge_categories_into_clients(categories, num_clients)

    clients: Dict[int, Dict[str, List[str]]] = {}
    for client_id, texts in enumerate(texts_bins):
        rng.shuffle(texts)
        n_eval = int(len(texts) * eval_fraction)
        eval_texts = texts[:n_eval]
        train_texts = texts[n_eval:]
        clients[client_id] = {"train": train_texts, "eval": eval_texts}
        category_label = ", ".join(sorted(category_bins[client_id])) or "none"
        print(
            f"[data] client {client_id}: {len(train_texts)} train / "
            f"{len(eval_texts)} eval (categories: {category_label})"
        )
    return clients
