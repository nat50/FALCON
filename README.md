# FALCON

**Federated Agentic LoRA for Constrained Optimization Networks**
---

## Architecture

<p align="center">
  <img src="images/architecture_diagram.png" alt="FALCON Architecture" width="900">
</p>

FALCON has three components:

| Component | Role |
|---|---|
| **Client** | Fine-tunes local LoRA adapters on private data. Always uploads `A`; uploads `B` only if selected. |
| **Server** | Builds a consensus subspace from all `A` matrices via SVD, aggregates the selected `B`-clients' updates, and factorizes the result into global adapters `(A_g, B_g)`. |
| **Agentic Control Plane** | Computes dynamic rank scores and solves an exact 0/1 knapsack, maximizing total alignment subject to the round's communication budget, to choose which clients upload `B` next round. |

Baselines (FedIT, FedSA, FlexLoRA) do not use the agent or dynamic ranks; they use fixed or data-ranked ranks and are included for comparison (see [Running Baselines](#running-baselines)).

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Verify the core merge math (no GPU required)
python scripts/check_merge.py

# 2. Run the full FALCON pipeline (requires GPU)
python main.py
```

Each run writes to a timestamped folder under `output/`:

| File | Contents |
|---|---|
| `run.log` | Full console log |
| `config.json` | Exact configuration used |
| `metrics.csv` | Per-round: mean eval loss, communication cost, number of `B`-uploaders |
| `per_client_eval.csv` | Per-round, per-client eval loss |
| `summary.json` | Final eval loss and total communication cost |

---

## Configuration

All settings live in [`falcon/config.py`](falcon/config.py). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `baseline_method` | `"falcon"` | Strategy to run: `"falcon"`, `"fedit"`, `"fedsa"`, `"flexlora"` |
| `client_model_name` | `"Qwen/Qwen3-0.6B-Base"` | Base model for LoRA fine-tuning |
| `lora_target_modules` | `[q, k, v, o_proj]` | Self-attention modules to attach LoRA |
| `lora_dropout` | `0.05` | LoRA dropout rate |
| `rank_pool` | `[4, 8, 16, 32]` | Discrete rank pool for FALCON; single-value list for fixed-rank baselines |
| `rank_alpha` / `rank_beta` / `rank_gamma` | `0.3` / `0.3` / `0.4` | Rank-score weights: data scale, learning difficulty, consensus novelty |
| `num_clients` | `8` | Number of federated clients |
| `num_rounds` | `20` | Communication rounds (`T`) |
| `local_epochs` | `1` | Local training epochs per round |
| `local_lr` | `2e-4` | AdamW learning rate |
| `max_seq_len` | `1024` | Maximum sequence length |
| `b_budget_fraction` | `0.4` | Fraction of total rank capacity allocated to `B`-uploads (`f_B`) |
| `data_path` | `"databricks/databricks-dolly-15k"` | Hugging Face dataset path |
| `eval_fraction` | `0.1` | Hold-out fraction for local evaluation |
| `num_gpus_per_client` | `1.0` | GPU fraction per simulated client (`0` for CPU-only) |

---

## Running Baselines

Switch between methods by changing `baseline_method` in [`falcon/config.py`](falcon/config.py):

```python
# FALCON (default): dynamic rank + selective B + knapsack agent
baseline_method = "falcon"
rank_pool = [4, 8, 16, 32]
b_budget_fraction = 0.4          # try 0.1, 0.4, 0.8

# FedIT: fixed rank, all clients upload A+B every round
baseline_method = "fedit"
rank_pool = [8]                   # single fixed rank

# FedSA: fixed rank, only A is shared, B stays local
baseline_method = "fedsa"
rank_pool = [16]                  # single fixed rank

# FlexLoRA: heterogeneous ranks, all clients upload A+B
baseline_method = "flexlora"
rank_pool = [2, 4, 6, 8, 10, 12, 14, 16]  # one rank per client
```

Then run:

```bash
python main.py
```

---

## Results

Experiments fine-tune `Qwen/Qwen3-0.6B-Base` on Databricks Dolly-15k, partitioned by task category across 8 non-IID clients, for 20 communication rounds. Full experimental setup and discussion are in the paper (Section 4).

### Communication-Performance Trade-off

Final evaluation loss and cumulative communication cost after 20 rounds:

| Method | Final Loss (L_eval) | Comm. Cost | Avg. B Uploads / Round |
|---|---|---|---|
| FedIT (r = 8) | 0.3716 | 2560 | 8.00 |
| FedSA (r = 16) | 0.4741 | 2560 | 0.00 |
| FlexLoRA | 0.3715 | 2880 | 8.00 |
| **FALCON** (f_B = 0.1) | 0.3820 | **1292** | 1.05 |
| **FALCON** (f_B = 0.4) | 0.3777 | 1708 | 3.35 |
| **FALCON** (f_B = 0.8) | 0.3723 | 2136 | 6.05 |

At `f_B = 0.4`, FALCON uses 33.28% less cumulative communication than FedIT and FedSA, and 40.69% less than FlexLoRA, while keeping the final evaluation loss within 0.01 of the strongest baseline (FlexLoRA) and clearly below FedSA.

### Evaluation Loss Over Communication Rounds

<p align="center">
  <img src="images/loss_curves.png" alt="Mean evaluation loss over communication rounds" width="900">
</p>

Mean evaluation loss over communication rounds under sample-proportional aggregation. A logarithmic y-axis shows both the warmup rounds and the post-warmup loss range in a single plot.
