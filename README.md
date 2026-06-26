# FALCON

**F**eder**A**ted **L**oRA with **C**lient-selective uploading via an **O**rchestratio**N** agent.

Federated fine-tuning of an LLM with LoRA where a **server-side agent** decides, every
round and per client, whether the client uploads only its **A** matrix (cheap, keeps the
client's knowledge private) or **both A and B** (shares full common knowledge). The server
merges all A's into a consensus subspace and the selected clients' B's into the shared
content, handling **heterogeneous LoRA ranks** across clients.

The full algorithm and all formulas are in [`docs/THUAT_TOAN.md`](docs/THUAT_TOAN.md) (Vietnamese).

## Components

| File | Role |
| --- | --- |
| `falcon/config.py` | All hyper-parameters in one place |
| `falcon/lora_math.py` | Consensus subspace, weighted merge, SVD factorize, rank truncation (pure NumPy) |
| `falcon/data.py` | Non-IID clients from `databricks-dolly-15k` (one category per client) |
| `falcon/modeling.py` | Qwen + LoRA: build, read/write A & B, train, evaluate |
| `falcon/agent.py` | Selection agent: Gemma GGUF (text-only) with a heuristic fallback |
| `falcon/client.py` | Flower client (adopt shared A, keep personal B, train, upload) |
| `falcon/strategy.py` | Flower strategy (the server-side FALCON logic) |
| `falcon/state_store.py` | On-disk store for each client's personal B |
| `main.py` | Runs the Flower simulation |
| `scripts/check_merge.py` | NumPy-only sanity check of the merge math |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

`llama-cpp-python` (the Gemma agent) is optional: if it is not installed, the system
automatically uses the heuristic agent. To use the LLM agent, set
`agent_kind = "llm"` in `falcon/config.py`.

## Run

```bash
# 1) Verify the core math (no heavy deps):
python scripts/check_merge.py

# 2) Run the federated simulation (single mode set in config):
python main.py

# 3) Run all baselines and print a comparison table:
python run_experiments.py
```

### Baselines (one pipeline, switch `selection_mode` in the config)

| mode | B-upload policy | meaning |
| --- | --- | --- |
| `fedsa` | nobody | share consensus A only, keep B local (FedSA-LoRA) |
| `flexlora` | everybody | full aggregation every round (FlexLoRA-style) |
| `falcon` | agent picks under budget | ours |

`run_experiments.py` runs the three on the same data and reports
**final eval loss** vs **total communication cost** — the core table/figure for the paper.

## Notes / assumptions

- **Client model**: defaults to `Qwen/Qwen2.5-0.5B-Instruct` (a known-good small Qwen).
  The original request mentioned "Qwen 3.5 0.8B"; change `client_model_name` in the config
  to your exact checkpoint id.
- **Agent model**: `google/gemma-4-E2B-it-qat-q4_0-gguf`, used text-only for the selection
  decision (no multimodal input).
- Defaults are sized for a quick CPU smoke test (6 clients, capped data). Increase
  `max_train_per_client`, `num_rounds`, and set `device = "cuda"` for real experiments.

## Suggested benchmarks (for the paper)

- **Personalization**: per-client test loss / perplexity.
- **Common knowledge**: loss on a mixed held-out test set (`data.build_global_testset`).
- **Communication**: `comm_cost` logged each round.
- **Main figure**: accuracy/loss vs communication cost, compared against
  FedSA-LoRA (A only, always) and FlexLoRA / HETLoRA (A and B, always).
