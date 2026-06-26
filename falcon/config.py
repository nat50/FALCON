"""Central configuration for a FALCON run.

All tunable knobs live here so the rest of the code stays free of magic numbers.
Keep this file flat and readable; do not add logic here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- Client model (LoRA fine-tuning target) -----
    client_model_name: str = "Qwen/Qwen3-0.6B-Base"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # ----- Heterogeneous ranks -----
    # Each client is assigned a rank from this pool (cycled by client index).
    client_rank_pool: List[int] = field(default_factory=lambda: [4, 8, 16])
    global_rank: int = 16  # rank R of the server-side global adapter

    # ----- Federation -----
    num_clients: int = 6
    num_rounds: int = 5
    local_epochs: int = 1
    local_batch_size: int = 4
    local_lr: float = 2e-4
    max_seq_len: int = 512
    freeze_shared_A: bool = False

    # ----- Selection mode (baseline switch) -----
    # "fedsa"   : nobody uploads B  -> share consensus A only, keep B local (FedSA-LoRA).
    # "flexlora": everybody uploads B -> full aggregation every round (FlexLoRA-style).
    # "falcon"  : the agent picks who uploads B under the budget (ours).
    selection_mode: str = "falcon"

    # ----- Agent / selection budget -----
    # Communication budget for uploading B, expressed as a fraction of the
    # cost of "all clients upload B". 1.0 means no limit; 0.4 means ~40%.
    b_budget_fraction: float = 0.4
    # Gemma GGUF used by the LLM agent (text-only usage).
    agent_repo_id: str = "google/gemma-4-E2B-it-qat-q4_0-gguf"
    agent_gguf_filename: str = "gemma-4-E2B_q4_0-it.gguf"
    agent_n_ctx: int = 4096

    # ----- Data -----
    dataset_name: str = "databricks/databricks-dolly-15k"
    max_train_per_client: int = 200  # cap for fast runs; raise for real experiments
    max_test_per_client: int = 40

    # ----- Misc -----
    seed: int = 42
    state_dir: str = "./client_state"  # where clients persist their personal B
    device: str = "cpu"  # "cuda" if a GPU is available


def default_config() -> Config:
    return Config()
