"""Non-IID client data for instruction-tuning federated experiments.

Supported datasets:
  - databricks/databricks-dolly-15k: clients are Dolly categories.
  - HuggingFaceH4/no_robots: clients are No Robots categories.
  - allenai/tulu-v2-sft-mixture: clients are source datasets in the mixture.

Each client returns plain lists of formatted text strings, so the client trainer
stays simple and framework-agnostic.
"""

import random
from typing import Callable, Dict, Iterable, List, Tuple

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

SUPPORTED_DATASETS = {
    "databricks/databricks-dolly-15k",
    "HuggingFaceH4/no_robots",
    "allenai/tulu-v2-sft-mixture",
}


def format_dolly_example(example: dict) -> str:
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


def format_messages(messages: List[dict]) -> str:
    """Render chat-style rows into a single causal-LM training string."""
    rendered = []
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    for message in messages:
        role = role_names.get(str(message.get("role", "")).lower(), "Message")
        content = str(message.get("content", "")).strip()
        if content:
            rendered.append(f"### {role}:\n{content}")
    return "\n\n".join(rendered)


def format_no_robots_example(example: dict) -> str:
    """Render one No Robots row into a single training string."""
    text = format_messages(example.get("messages") or [])
    if text:
        return text
    return PROMPT_NO_CONTEXT.format(
        instruction=str(example.get("prompt", "")).strip(),
        response="",
    )


def format_tulu_example(example: dict) -> str:
    """Render one Tulu SFT mixture row into a single training string."""
    return format_messages(example.get("messages") or [])


def _dataset_group_and_formatter(
    dataset_name: str,
) -> Tuple[str, Callable[[dict], str]]:
    """Return the grouping column and row formatter for a supported dataset."""
    if dataset_name == "databricks/databricks-dolly-15k":
        return "category", format_dolly_example
    if dataset_name == "HuggingFaceH4/no_robots":
        return "category", format_no_robots_example
    if dataset_name == "allenai/tulu-v2-sft-mixture":
        return "dataset", format_tulu_example
    supported = ", ".join(sorted(SUPPORTED_DATASETS))
    raise ValueError(f"unsupported dataset '{dataset_name}'. Supported: {supported}")


def _group_examples(
    rows: Iterable[dict],
    group_field: str,
    formatter: Callable[[dict], str],
) -> Dict[str, List[str]]:
    """Group formatted examples by non-IID client key."""
    grouped: Dict[str, List[str]] = {}
    skipped = 0
    for row in rows:
        group = str(row.get(group_field, "")).strip()
        text = formatter(row).strip()
        if not group or not text:
            skipped += 1
            continue
        grouped.setdefault(group, []).append(text)
    if skipped:
        print(f"[data] skipped {skipped} rows with missing group/text")
    return grouped


def load_client_datasets(
    dataset_name: str,
    num_clients: int,
    max_train: int,
    max_test: int,
    seed: int,
) -> Dict[int, Dict[str, List[str]]]:
    """Group a supported instruction dataset into non-IID client splits.

    Returns:
        {client_id: {"train": [str, ...], "test": [str, ...], "category": str}}
    """
    rng = random.Random(seed)
    group_field, formatter = _dataset_group_and_formatter(dataset_name)
    dataset = load_dataset(dataset_name, split="train")

    by_group = _group_examples(dataset, group_field, formatter)
    if not by_group:
        raise ValueError(f"dataset '{dataset_name}' produced no usable examples")

    # Pick the largest groups so every client has enough data.
    groups = sorted(by_group, key=lambda group: len(by_group[group]), reverse=True)
    groups = groups[:num_clients]
    if len(groups) < num_clients:
        print(
            f"[data] requested {num_clients} clients but only found "
            f"{len(groups)} groups in '{dataset_name}'"
        )

    clients: Dict[int, Dict[str, List[str]]] = {}
    for client_id, group in enumerate(groups):
        texts = by_group[group]
        rng.shuffle(texts)
        test = texts[:max_test]
        train = texts[max_test : max_test + max_train]
        clients[client_id] = {"train": train, "test": test, "category": group}
        print(
            f"[data] client {client_id} <- {group_field} '{group}': "
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
