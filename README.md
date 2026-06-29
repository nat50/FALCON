# FALCON

**F**ederated-**A**gentic-**L**oRA-for-**C**onstrained-**O**ptimization-**N**etworks.

Federated fine-tuning of an LLM with LoRA where a server-side agent decides, every
round and per client, whether the client uploads only its **A** matrix (cheap, keeps
the client's knowledge private) or **both A and B** (shares full common knowledge).
The server merges all A's into a consensus subspace and the selected clients' B's into
the shared content, handling heterogeneous LoRA ranks across clients.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

`llama-cpp-python` (the Gemma agent) is required: the selection agent is always the
LLMAgent and there is no heuristic fallback. Configure the model via `agent_repo_id` /
`agent_gguf_filename` in `falcon/config.py`.

## Dataset

The pipeline trains on **Fed-WildChat** from
[FedLLM-Bench](https://github.com/rui-ye/FedLLM-Bench): real human-chatbot conversations
naturally partitioned by user. Download the Fed-WildChat data from the FedLLM-Bench
data link and keep the single-turn split at
`data/FedLLM-Bench-Data/Fed-WildChat/single_turn/wildchat_100c_53k.json`, a JSON
object mapping each user id to a list of `{"instruction", "response"}` samples.

The loader uses the full dataset, merges the users into `num_clients` clients balanced by
sample count, and holds out `eval_fraction` of each client for evaluation.

## Run

```bash
# 1) Verify the core merge math (no heavy deps):
python scripts/check_merge.py

# 2) Run the full FALCON pipeline:
python main.py
```

## Output

Each run writes a timestamped folder under `output/` containing:

| File | Contents |
| --- | --- |
| `run.log` | Full console log of the run |
| `config.json` | The exact config used |
| `metrics.csv` | Per-round mean eval loss, communication cost, number of B-uploaders |
| `per_client_eval.csv` | Per-round, per-client eval loss |
| `summary.json` | Final eval loss and total communication cost |

## Configuration

All tunable settings live in `falcon/config.py`, including the client model, LoRA ranks,
number of clients and rounds, sequence length, communication budget, and GPU allocation
per simulated client.
