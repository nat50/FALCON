"""Central configuration for a FALCON run.

All tunable knobs live here so the rest of the code stays free of magic numbers.
Keep this file flat and readable; do not add logic here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- Method -----
    # "falcon" keeps the original dynamic-rank + selective-B strategy.
    # "fedsa" runs the FedSA baseline: fixed-rank, A-only aggregation, no agent.
    # "flexlora" runs the FlexLoRA baseline: data-ranked clients, full A+B upload.
    # "fedit" runs the FedIT baseline: fixed-rank, full A+B FedAvg, no agent.
    baseline_method: str = "flexlora"

    # ----- Client model (LoRA fine-tuning target) -----
    client_model_name: str = "Qwen/Qwen3-0.6B-Base"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05

    # ----- Heterogeneous ranks -----
    # Discrete ranks for initial assignment and per-round dynamic allocation.
    # LoRA alpha equals each client's rank (scale factor alpha/r = 1).
    rank_pool: List[int] = field(default_factory=lambda: [8])
    flexlora_rank_pool: List[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 10, 12, 14, 16]
    )
    rank_alpha: float = 0.3
    rank_beta: float = 0.3
    rank_gamma: float = 0.4

    # ----- Federation -----
    num_clients: int = 8
    num_rounds: int = 20
    local_epochs: int = 1
    local_batch_size: int = 8
    local_lr: float = 2e-4
    max_seq_len: int = 1024

    # ----- Agent / selection budget -----
    # Communication budget for uploading B, expressed as a fraction of the
    # cost of "all clients upload B". 1.0 means no limit; 0.4 means ~40%.
    b_budget_fraction: float = 0.4
    # Gemma GGUF used by the LLM agent (text-only usage).
    agent_repo_id: str = "google/gemma-4-E2B-it-qat-q4_0-gguf"
    agent_gguf_filename: str = "gemma-4-E2B_q4_0-it.gguf"
    agent_n_ctx: int = 4096

    # ----- Data -----
    data_path: str = "databricks/databricks-dolly-15k"
    eval_fraction: float = 0.1

    # ----- Flower simulation (Ray) -----
    # Fraction of a GPU per simulated client; set num_gpus_per_client=0 for CPU-only runs.
    num_cpus_per_client: int = 4
    num_gpus_per_client: float = 0.5

    # ----- Misc -----
    seed: int = 42
    state_dir: str = "./client_state"  # where clients persist their personal B
    output_dir: str = "./output"
    device: str = "cuda"


def default_config() -> Config:
    return Config()
