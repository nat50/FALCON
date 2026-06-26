# FALCON

**F**ederated-**A**gentic-**L**oRA-for-**C**onstrained-**O**ptimization-**N**etworks.

Federated fine-tuning of an LLM with LoRA where a **server-side agent** decides, every
round and per client, whether the client uploads only its **A** matrix (cheap, keeps the
client's knowledge private) or **both A and B** (shares full common knowledge). The server
merges all A's into a consensus subspace and the selected clients' B's into the shared
content, handling **heterogeneous LoRA ranks** across clients.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

`llama-cpp-python` (the Gemma agent) is **required**: the selection agent is always the
LLMAgent and there is no heuristic fallback. If the model cannot be loaded or parsed,
the run fails with a clear error. Configure the model via `agent_repo_id` /
`agent_gguf_filename` in `falcon/config.py`.

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
- Defaults are sized for a quick smoke test (6 clients, capped data). Increase
  `max_train_per_client`, `num_rounds`, and set `device = "cuda"` for real experiments.
