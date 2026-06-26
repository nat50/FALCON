"""Model + LoRA helpers: build, read/write LoRA factors, train, evaluate.

Keeps all PyTorch/PEFT details in one place so the Flower client stays thin.
LoRA naming in PEFT (per target module):
    ...<module>.lora_A.default.weight  -> shape (r, in_features)   == our A (r, k)
    ...<module>.lora_B.default.weight  -> shape (out_features, r)  == our B (d, r)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_A_SUFFIX = ".lora_A.default.weight"
_B_SUFFIX = ".lora_B.default.weight"


def build_model(model_name: str, rank: int, target_modules: List[str],
                alpha: int, dropout: float, device: str):
    """Load the base causal LM and wrap it with a LoRA adapter of the given rank."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    model.to(device)
    return model, tokenizer


def _module_key(param_name: str) -> str:
    """Strip the LoRA suffix to get a key shared by the A and B of one module."""
    return param_name.replace(_A_SUFFIX, "").replace(_B_SUFFIX, "")


def get_lora_AB(model) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {module_key: (A (r, k), B (d, r))} as detached NumPy arrays."""
    a_parts: Dict[str, np.ndarray] = {}
    b_parts: Dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if name.endswith(_A_SUFFIX):
            a_parts[_module_key(name)] = param.detach().cpu().numpy()
        elif name.endswith(_B_SUFFIX):
            b_parts[_module_key(name)] = param.detach().cpu().numpy()
    return {key: (a_parts[key], b_parts[key]) for key in a_parts}


def set_lora_factor(model, key: str, which: str, value: np.ndarray) -> None:
    """Write a single A or B matrix back into the model in-place. `which` is 'A' or 'B'."""
    suffix = _A_SUFFIX if which == "A" else _B_SUFFIX
    target = key + suffix
    for name, param in model.named_parameters():
        if name == target:
            tensor = torch.tensor(value, dtype=param.dtype, device=param.device)
            with torch.no_grad():
                param.copy_(tensor)
            return
    raise KeyError(f"LoRA parameter not found: {target}")


def freeze_A(model) -> None:
    """Disable gradients on every LoRA A matrix (used when freeze_shared_A=True)."""
    for name, param in model.named_parameters():
        if name.endswith(_A_SUFFIX):
            param.requires_grad = False


def _encode(tokenizer, texts: List[str], max_len: int):
    batch = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )
    batch["labels"] = batch["input_ids"].clone()
    return batch


def train_local(model, tokenizer, texts: List[str], lr: float, epochs: int,
                batch_size: int, max_len: int, device: str) -> float:
    """Fine-tune the LoRA params on local texts. Returns the mean training loss."""
    if not texts:
        return 0.0
    encoded = _encode(tokenizer, texts, max_len)
    loader = DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    total_loss, num_steps = 0.0, 0
    for _ in range(epochs):
        for index_batch in loader:
            idx = index_batch.tolist()
            inputs = {k: v[idx].to(device) for k, v in encoded.items()}
            optimizer.zero_grad()
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_steps += 1
    return total_loss / max(num_steps, 1)


@torch.no_grad()
def eval_loss(model, tokenizer, texts: List[str], batch_size: int,
              max_len: int, device: str) -> float:
    """Mean next-token loss on a set of texts (lower is better)."""
    if not texts:
        return 0.0
    encoded = _encode(tokenizer, texts, max_len)
    model.eval()
    total_loss, num_steps = 0.0, 0
    for start in range(0, len(texts), batch_size):
        idx = list(range(start, min(start + batch_size, len(texts))))
        inputs = {k: v[idx].to(device) for k, v in encoded.items()}
        total_loss += float(model(**inputs).loss.item())
        num_steps += 1
    return total_loss / max(num_steps, 1)
