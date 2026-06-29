# Guideline chạy baseline FlexLoRA riêng để so sánh với FALCON

Tài liệu này dùng khi **chạy baseline trong một repo riêng** (ví dụ repo FlexLoRA chính thức) nhưng vẫn muốn kết quả có thể so sánh công bằng với FALCON hiện tại.

Mục tiêu không phải là chạy y nguyên setup gốc của FlexLoRA, mà là:

> Giữ thuật toán baseline của FlexLoRA, nhưng ép **dataset, model, training protocol, metric, logging output** giống FALCON nhất có thể.

Nếu không làm vậy, kết quả sẽ khó so sánh vì khác biệt có thể đến từ dataset split, model, tokenizer, optimizer, số round, eval metric hoặc cách tính communication cost, không phải từ thuật toán.

---

## 1. Repo baseline cần dùng

Repo FlexLoRA chính thức:

```text
https://github.com/alibaba/FederatedScope/tree/FlexLoRA
```

Paper:

```text
Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources
https://proceedings.neurips.cc/paper_files/paper/2024/file/1a134b50202088aa8c595cc99b310e5a-Paper-Conference.pdf
```

Không dùng repo sau làm FlexLoRA baseline:

```text
https://github.com/Chongjie-Si/Subspace-Tuning
```

Repo `Subspace-Tuning` là framework PEFT/subspace tuning tổng quát. Nó có `FLoRA` nhưng không phải `FlexLoRA` federated baseline cần so sánh ở đây.

---

## 2. Nguyên tắc so sánh

Baseline chạy repo riêng vẫn phải tuân thủ các nguyên tắc sau:

1. **Cùng dữ liệu**: dùng đúng file `data/fed_wildchat.json` như FALCON.
2. **Cùng split**: train/eval split phải giống logic `falcon/data.py`.
3. **Cùng model nền**: dùng `Qwen/Qwen3-0.6B-Base`.
4. **Cùng LoRA target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
5. **Cùng số client**: mặc định 10 client.
6. **Cùng số round**: mặc định 30 round.
7. **Cùng local training**: local epochs, batch size, learning rate, max sequence length phải match FALCON.
8. **Cùng metric chính**: final mean eval loss và total communication cost.
9. **Cùng output schema**: output phải có `summary.json`, `metrics.csv`, `per_client_eval.csv`, `config.json`, `run.log`.
10. **Không dùng số từ paper gốc làm main result** nếu setup khác.

Trong paper nên ghi rõ:

```text
We run FlexLoRA in a separate codebase but enforce the same dataset split,
model, training protocol, evaluation metric, and output schema as FALCON.
```

---

## 3. FALCON config cần match

Các giá trị này lấy từ code FALCON hiện tại trong `falcon/config.py`.

```text
client_model_name = "Qwen/Qwen3-0.6B-Base"
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_dropout = 0.05

rank_pool = [4, 8, 16, 32]

num_clients = 10
num_rounds = 30
local_epochs = 2
local_batch_size = 8
local_lr = 2e-4
max_seq_len = 1024

data_path = "./data/fed_wildchat.json"
eval_fraction = 0.1

seed = 42
device = "cuda"
```

Nếu repo FlexLoRA không hỗ trợ đúng một tham số nào đó, phải ghi rõ trong `config.json` và trong note kết quả.

---

## 4. Dataset bắt buộc

FALCON hiện tại dùng Fed-WildChat từ FedLLM-Bench, lưu local:

```text
./data/fed_wildchat.json
```

Format file:

```json
{
  "user_id_1": [
    {"instruction": "...", "response": "..."},
    {"instruction": "...", "response": "..."}
  ],
  "user_id_2": [
    {"instruction": "...", "response": "..."}
  ]
}
```

Mỗi sample phải được format đúng như FALCON:

```text
### Instruction:
{instruction}

### Response:
{response}
```

Không thêm context, system prompt, chat template khác, hoặc role format khác nếu FALCON không dùng.

---

## 5. Quy tắc split client giống FALCON

Repo FlexLoRA phải implement cùng logic với `falcon/data.py`.

### 5.1. Load users

Đọc JSON thành:

```python
users: Dict[str, List[str]]
```

Mỗi `user_id` giữ nguyên toàn bộ sample của user đó. Không được trộn sample của một user trước khi gán client.

### 5.2. Merge users thành client

Logic bắt buộc:

```python
texts_bins = [[] for _ in range(num_clients)]
sizes = [0] * num_clients
user_counts = [0] * num_clients

ordered = sorted(users.values(), key=len, reverse=True)
for texts in ordered:
    target = min(range(num_clients), key=lambda i: sizes[i])
    texts_bins[target].extend(texts)
    sizes[target] += len(texts)
    user_counts[target] += 1
```

Ý nghĩa:

- Sắp user theo số sample giảm dần.
- Mỗi user được đưa vào client hiện có ít sample nhất.
- Mục tiêu là cân bằng số sample giữa client nhưng vẫn giữ ranh giới user.

### 5.3. Train/eval split

Sau khi merge:

```python
rng = random.Random(seed)
for each client:
    rng.shuffle(texts)
    n_eval = int(len(texts) * eval_fraction)
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]
```

Phải dùng `seed = 42` mặc định, trừ khi chạy multi-seed.

Output log nên in:

```text
[data] client {client_id}: {num_train} train / {num_eval} eval ({num_users} users)
```

---

## 6. Model và tokenizer

Baseline phải dùng cùng model:

```text
Qwen/Qwen3-0.6B-Base
```

Tokenizer:

- Nếu tokenizer không có `pad_token`, set `pad_token = eos_token`.
- Tokenize với `truncation=True`, `padding="max_length"`, `max_length=1024`.
- Labels bằng `input_ids.clone()`.

Tương đương logic FALCON:

```python
batch = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=max_seq_len,
    return_tensors="pt",
)
batch["labels"] = batch["input_ids"].clone()
```

Không dùng packing, response-only loss, chat template, hoặc masking khác nếu FALCON chưa dùng.

---

## 7. LoRA config

Baseline phải dùng LoRA trên cùng module:

```text
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
dropout = 0.05
task_type = "CAUSAL_LM"
```

Rank:

```text
rank_pool = [4, 8, 16, 32]
```

Initial rank assignment phải giống FALCON:

```python
client_ranks = {
    cid: rank_pool[cid % len(rank_pool)]
    for cid in range(num_clients)
}
```

Với 10 clients:

```text
client 0: 4
client 1: 8
client 2: 16
client 3: 32
client 4: 4
client 5: 8
client 6: 16
client 7: 32
client 8: 4
client 9: 8
```

Nếu FlexLoRA repo có rank distribution riêng (`normal`, `random`, `heavy_tail`, v.v.), tắt hoặc override để dùng đúng assignment trên.

---

## 8. Training protocol

Phải match FALCON:

```text
num_rounds = 30
local_epochs = 2
local_batch_size = 8
local_lr = 2e-4
max_seq_len = 1024
optimizer = AdamW
```

Local train loss:

- Causal LM next-token loss.
- Train trên toàn bộ `train_texts` của client.
- Eval loss trước/sau local training nếu cần logging, nhưng metric chính là eval set sau round.

Nếu repo FlexLoRA dùng local steps thay vì local epochs:

- Cần quy đổi hoặc sửa sang local epochs.
- Nếu không sửa được, ghi rõ khác biệt trong `config.json`.
- Không nên so sánh main result nếu local compute khác quá nhiều.

---

## 9. Thuật toán FlexLoRA cần giữ

Khi chạy FlexLoRA baseline, cần giữ đúng ý tưởng cốt lõi của FlexLoRA:

1. Mỗi client train LoRA với rank riêng `r_i`.
2. Client upload đầy đủ LoRA factors `A_i`, `B_i`.
3. Server dựng full update:

```math
\Delta W_i = B_i A_i
```

4. Server average theo số sample:

```math
\Delta W_g =
\frac{\sum_i n_i \Delta W_i}{\sum_i n_i}
```

5. Server SVD:

```math
\Delta W_g = U \Sigma V^\top
```

6. Server redistribute cho client theo rank:

```math
A_i^{new} = \Sigma_{r_i}^{1/2} V_{r_i}^\top
B_i^{new} = U_{r_i} \Sigma_{r_i}^{1/2}
```

Trong FALCON, proposed method chỉ upload B từ selected clients. Còn FlexLoRA baseline phải upload A+B từ tất cả clients mỗi round.

Không dùng agent, không dùng budget, không dùng rank-score weighting của FALCON.

---

## 10. Những thành phần FALCON không được đưa vào FlexLoRA baseline

Để baseline sạch, repo FlexLoRA không được dùng:

- LLM agent chọn client.
- Budgeted B upload.
- Alignment-based client selection.
- FALCON rank score:

```math
S_i = \alpha S_i^{data} + \beta S_i^{learn} + \gamma S_i^{novelty}
```

- Rank-score aggregation weight `S_i * n_i`.
- Personal B local-only logic của FALCON.
- Partial B upload.

FlexLoRA baseline phải là full upload baseline:

```text
all clients upload A+B every round
```

---

## 11. Communication cost phải xuất giống FALCON

FALCON hiện mô hình hóa communication cost theo rank:

```math
comm_t = \sum_i r_i + \sum_{i \in S_t} r_i
```

Với FlexLoRA full upload:

```math
S_t = N
```

nên:

```math
comm_t^{FlexLoRA} = \sum_i r_i + \sum_i r_i = 2 \sum_i r_i
```

Nếu rank cố định theo assignment ban đầu:

```text
ranks = [4, 8, 16, 32, 4, 8, 16, 32, 4, 8]
sum_ranks = 132
comm_per_round = 264
total_comm_cost over 30 rounds = 7920
```

Nếu FlexLoRA có dynamic redistribution nhưng rank capacity không đổi, vẫn tính theo rank capacity của client.

Không tính bytes thực tế trong main comparison nếu FALCON đang tính rank-unit cost. Nếu muốn báo cáo bytes thực tế, thêm metric phụ riêng, không thay thế metric chính.

---

## 12. Evaluation metric phải giống FALCON

Mỗi round, evaluate trên eval set riêng của từng client:

```python
loss_i = eval_loss(model_i, eval_texts_i)
```

Mean eval loss:

```math
mean\_eval\_loss =
\frac{\sum_i |\mathcal{D}_i^{eval}| loss_i}
{\sum_i |\mathcal{D}_i^{eval}|}
```

Phải lưu cả:

- mean eval loss mỗi round.
- eval loss từng client mỗi round.
- final eval loss ở round cuối.
- total communication cost.

Nếu repo FlexLoRA chỉ eval global model trên benchmark ngoài, cần sửa. Không dùng benchmark ngoài làm metric chính.

---

## 13. Output schema bắt buộc

Mỗi run của repo FlexLoRA nên tạo:

```text
output/
  run_YYYYMMDD_HHMMSS/
    run.log
    config.json
    metrics.csv
    per_client_eval.csv
    summary.json
```

### 13.1. `config.json`

Phải chứa tối thiểu:

```json
{
  "method": "flexlora",
  "client_model_name": "Qwen/Qwen3-0.6B-Base",
  "data_path": "./data/fed_wildchat.json",
  "num_clients": 10,
  "num_rounds": 30,
  "local_epochs": 2,
  "local_batch_size": 8,
  "local_lr": 0.0002,
  "max_seq_len": 1024,
  "rank_pool": [4, 8, 16, 32],
  "client_ranks": {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32,
    "4": 4,
    "5": 8,
    "6": 16,
    "7": 32,
    "8": 4,
    "9": 8
  },
  "seed": 42
}
```

Nếu có tham số FlexLoRA riêng, thêm vào nhưng không được thiếu các field trên.

### 13.2. `metrics.csv`

Columns bắt buộc:

```text
round,mean_eval_loss,comm_cost,num_b_uploaders
```

Với FlexLoRA:

```text
num_b_uploaders = num_clients
```

Ví dụ:

```csv
round,mean_eval_loss,comm_cost,num_b_uploaders
1,2.9134,264,10
2,2.7741,264,10
...
30,2.1029,264,10
```

### 13.3. `per_client_eval.csv`

Columns bắt buộc:

```text
round,client_id,eval_loss
```

Ví dụ:

```csv
round,client_id,eval_loss
1,0,2.91
1,1,3.05
...
30,9,2.44
```

### 13.4. `summary.json`

Phải match FALCON:

```json
{
  "method": "flexlora",
  "num_clients": 10,
  "num_rounds": 30,
  "final_eval_loss": 2.1029,
  "total_comm_cost": 7920.0
}
```

Có thể thêm field phụ:

```json
{
  "best_eval_loss": 2.0981,
  "final_round": 30,
  "notes": "Same dataset split and output schema as FALCON."
}
```

---

## 14. Logging bắt buộc

`run.log` nên có các dòng tương tự:

```text
[output] writing run artifacts to output/run_YYYYMMDD_HHMMSS
[data] client 0: ... train / ... eval (... users)
[main] client ranks: {0: 4, 1: 8, ...}
[server] round 1: FlexLoRA full upload from all clients
[client 0] trained (loss_before=..., loss_after=..., sent_B=True, rank=4)
[server] round 1: merged (B from [0,1,2,3,4,5,6,7,8,9]), comm_cost=264
[server] round 1: mean eval loss = ...
```

Mục tiêu là đọc log có thể audit được:

- Dataset split đúng chưa.
- Rank đúng chưa.
- Mỗi round có đủ client upload B không.
- Communication cost đúng chưa.
- Eval loss đúng chưa.

---

## 15. Cách sửa repo FlexLoRA đề xuất

### 15.1. Thêm data loader tương thích FALCON

Tạo file mới, ví dụ:

```text
data_fed_wildchat.py
```

Chứa các hàm:

```python
format_example(example)
load_users(data_path)
merge_users_into_clients(users, num_clients)
load_client_datasets(data_path, num_clients, eval_fraction, seed)
```

Copy logic từ `falcon/data.py`.

### 15.2. Thêm config adapter

Tạo config riêng:

```text
configs/falcon_compare_flexlora.yaml
```

hoặc file Python/JSON tương đương.

Config này phải set các tham số ở Mục 3.

### 15.3. Sửa model builder

Đảm bảo model builder hỗ trợ:

```text
Qwen/Qwen3-0.6B-Base
target_modules = q_proj,k_proj,v_proj,o_proj
```

Nếu repo FlexLoRA dùng `loralib` custom thay vì PEFT:

- Cần map đúng tên module.
- Cần đảm bảo đọc/ghi được factor `A`, `B` cùng shape với FALCON:

```text
A: (r, in_features)
B: (out_features, r)
```

### 15.4. Sửa training loop

Đảm bảo local loop:

- dùng AdamW.
- dùng local epochs = 2.
- batch size = 8.
- full sequence loss giống FALCON.
- không response-only masking.

### 15.5. Sửa server aggregation FlexLoRA

Ở server, với mỗi LoRA layer:

```python
delta_bar = 0
weight_sum = sum(num_examples)
for client in clients:
    delta_i = B_i @ A_i
    delta_bar += (num_examples_i / weight_sum) * delta_i

U, S, Vt = svd(delta_bar)
for client:
    r = client_rank[client]
    sqrt_s = sqrt(S[:r])
    A_new = sqrt_s[:, None] * Vt[:r, :]
    B_new = U[:, :r] * sqrt_s[None, :]
```

Không thêm FALCON consensus projection `P = V^T V` nếu đang chạy đúng FlexLoRA baseline. FlexLoRA aggregate full update rồi SVD redistribute.

### 15.6. Sửa output writer

Tạo writer giống `falcon/results.py`:

```python
RunLogger(output_dir)
dump_config(config)
save_round_metrics(rows)
save_per_client_eval(rows)
save_summary(summary)
```

Nếu repo FlexLoRA đã có logger riêng, thêm export sang schema ở Mục 13.

---

## 16. Checklist trước khi chạy

Trước khi chạy baseline, kiểm tra:

- [ ] Dataset path là `./data/fed_wildchat.json`.
- [ ] Split client giống FALCON.
- [ ] Số client là 10.
- [ ] Seed là 42.
- [ ] Model là `Qwen/Qwen3-0.6B-Base`.
- [ ] Target modules là `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- [ ] Rank assignment là `[4,8,16,32,4,8,16,32,4,8]`.
- [ ] Num rounds là 30.
- [ ] Local epochs là 2.
- [ ] Batch size là 8.
- [ ] LR là `2e-4`.
- [ ] Max seq len là 1024.
- [ ] Mọi client upload A+B mỗi round.
- [ ] Không dùng agent.
- [ ] Không dùng budget.
- [ ] Không dùng rank-score weighting.
- [ ] `metrics.csv` có đủ 30 rows.
- [ ] `per_client_eval.csv` có `30 * 10 = 300` rows.
- [ ] `summary.json` có `final_eval_loss` và `total_comm_cost`.

---

## 17. Checklist sau khi chạy

Sau khi chạy xong, kiểm tra:

### 17.1. Communication cost

Nếu rank cố định như FALCON initial assignment:

```text
comm_cost mỗi round = 264
total_comm_cost = 7920
```

Nếu khác, phải giải thích vì sao.

### 17.2. Eval rows

Kiểm tra:

```text
metrics.csv: 30 dòng round
per_client_eval.csv: 300 dòng nếu 10 clients x 30 rounds
```

### 17.3. Config reproducibility

`config.json` phải đủ để chạy lại cùng kết quả:

- model
- dataset
- split seed
- rank assignment
- hyperparameters
- method name
- repo commit hash nếu có

Nên thêm:

```json
{
  "git_commit": "...",
  "baseline_repo": "alibaba/FederatedScope FlexLoRA branch"
}
```

---

## 18. Kết quả nên đem về FALCON repo như thế nào

Sau khi chạy repo FlexLoRA riêng, copy output folder về FALCON project:

```text
external_baselines/
  flexlora/
    run_YYYYMMDD_HHMMSS/
      run.log
      config.json
      metrics.csv
      per_client_eval.csv
      summary.json
```

Không copy checkpoint nặng nếu không cần.

Khi viết bảng so sánh:

```text
method      final_eval_loss    total_comm_cost
FALCON      ...                ...
FlexLoRA    ...                ...
FedSA       ...                ...
```

Chỉ so sánh các run có cùng protocol.

---

## 19. Cách ghi trong paper

Nếu chạy repo riêng nhưng ép cùng protocol, có thể viết:

```text
For FlexLoRA, we use the official implementation as a reference and adapt its
training/evaluation pipeline to match our experimental protocol: the same
Fed-WildChat split, base model, LoRA targets, number of clients, local training
configuration, evaluation loss, and communication-cost accounting.
```

Nếu có khác biệt không tránh được:

```text
We note that the FlexLoRA baseline is executed in a separate codebase; to ensure
comparability, we export results using the same output schema and report only
metrics computed under the same dataset split and evaluation protocol.
```

Không nên claim:

```text
We directly compare against the original reported FlexLoRA numbers.
```

nếu không cùng setup.

---

## 20. Kết luận khuyến nghị

Chạy baseline ở repo riêng là được, nhưng chỉ hợp lý nếu:

- output giống FALCON,
- dataset split giống FALCON,
- metric giống FALCON,
- communication cost giống FALCON,
- config/hyperparameter được log đầy đủ.

Nếu không đảm bảo các điều trên, kết quả chỉ nên dùng làm tham khảo, không dùng làm main comparison.

Ưu tiên thực tế:

1. Dùng repo FlexLoRA chính thức để đọc và verify thuật toán.
2. Nếu chạy riêng, sửa theo guideline này.
3. Nếu sửa quá nặng, port FlexLoRA aggregation vào FALCON pipeline sẽ công bằng và ít rủi ro hơn.

---

## Appendix A. FALCON reference snippets bắt buộc copy đúng

Phần này giúp coding agent ở repo FlexLoRA không cần mở repo FALCON vẫn có thể implement đúng protocol. Nếu có xung đột giữa mô tả ở trên và snippet bên dưới, ưu tiên snippet bên dưới.

### A.1. Dataset loader exact

Copy logic này sang repo FlexLoRA, có thể đổi tên file/hàm nhưng không đổi hành vi:

```python
import json
import random
from typing import Dict, List, Tuple

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)


def format_example(example: dict) -> str:
    """Render one sample into a single training string."""
    return PROMPT_TEMPLATE.format(
        instruction=str(example.get("instruction", "")).strip(),
        response=str(example.get("response", "")).strip(),
    )


def _load_users(data_path: str) -> Dict[str, List[str]]:
    """Read the raw JSON and return {user_id: [formatted_text, ...]}."""
    with open(data_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        user_id: [format_example(sample) for sample in samples]
        for user_id, samples in raw.items()
    }


def _merge_users_into_clients(
    users: Dict[str, List[str]], num_clients: int
) -> Tuple[List[List[str]], List[int]]:
    """Greedily assign whole users to clients, balancing total sample counts."""
    texts_bins: List[List[str]] = [[] for _ in range(num_clients)]
    user_counts = [0] * num_clients
    sizes = [0] * num_clients
    ordered = sorted(users.values(), key=len, reverse=True)
    for texts in ordered:
        target = min(range(num_clients), key=lambda i: sizes[i])
        texts_bins[target].extend(texts)
        sizes[target] += len(texts)
        user_counts[target] += 1
    return texts_bins, user_counts


def load_client_datasets(
    data_path: str,
    num_clients: int,
    eval_fraction: float,
    seed: int,
) -> Dict[int, Dict[str, List[str]]]:
    """Build per-client train/eval splits from the Fed-WildChat JSON file."""
    rng = random.Random(seed)
    users = _load_users(data_path)
    texts_bins, user_counts = _merge_users_into_clients(users, num_clients)

    clients: Dict[int, Dict[str, List[str]]] = {}
    for client_id, texts in enumerate(texts_bins):
        rng.shuffle(texts)
        n_eval = int(len(texts) * eval_fraction)
        eval_texts = texts[:n_eval]
        train_texts = texts[n_eval:]
        clients[client_id] = {"train": train_texts, "eval": eval_texts}
        print(
            f"[data] client {client_id}: {len(train_texts)} train / "
            f"{len(eval_texts)} eval ({user_counts[client_id]} users)"
        )
    return clients
```

### A.2. Rank assignment exact

```python
def assign_ranks(num_clients: int, rank_pool):
    """Give each client a (possibly different) LoRA rank from the pool."""
    return {cid: rank_pool[cid % len(rank_pool)] for cid in range(num_clients)}
```

Với default:

```python
rank_pool = [4, 8, 16, 32]
num_clients = 10
client_ranks = assign_ranks(num_clients, rank_pool)
```

Kết quả phải là:

```json
{
  "0": 4,
  "1": 8,
  "2": 16,
  "3": 32,
  "4": 4,
  "5": 8,
  "6": 16,
  "7": 32,
  "8": 4,
  "9": 8
}
```

### A.3. Tokenization exact

```python
def _encode(tokenizer, texts, max_len: int):
    batch = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    batch["labels"] = batch["input_ids"].clone()
    return batch
```

Không thêm:

- chat template,
- response-only masking,
- packing,
- loss mask cho instruction,
- EOS/prompt format khác.

### A.4. Model/LoRA builder exact target behavior

FALCON dùng PEFT:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def build_model(model_name, rank, target_modules, alpha, dropout, device):
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
```

Trong FlexLoRA repo có thể không dùng PEFT, nhưng phải đảm bảo tương đương:

```text
model_name = Qwen/Qwen3-0.6B-Base
target_modules = q_proj,k_proj,v_proj,o_proj
rank = client-specific rank
alpha = rank
dropout = 0.05
dtype = float32 nếu tài nguyên cho phép
```

Nếu bắt buộc dùng dtype khác do memory, phải ghi trong `config.json`.

### A.5. Train/eval loop exact target behavior

```python
from torch.utils.data import DataLoader
import torch


def train_local(model, tokenizer, texts, lr, epochs, batch_size, max_len, device):
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
def eval_loss(model, tokenizer, texts, batch_size, max_len, device):
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
```

### A.6. Output writer exact

Repo FlexLoRA nên có logger tương đương:

```python
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class RunLogger:
    def __init__(self, output_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self._log_file = open(self.path("run.log"), "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log_file)
        print(f"[output] writing run artifacts to {self.run_dir}")

    def path(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def dump_config_dict(self, config: Dict) -> None:
        with open(self.path("config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    def save_round_metrics(self, rows: List[Dict]) -> None:
        self._write_csv(
            "metrics.csv",
            ["round", "mean_eval_loss", "comm_cost", "num_b_uploaders"],
            rows,
        )

    def save_per_client_eval(self, rows: List[Dict]) -> None:
        self._write_csv(
            "per_client_eval.csv",
            ["round", "client_id", "eval_loss"],
            rows,
        )

    def save_summary(self, summary: Dict) -> None:
        with open(self.path("summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _write_csv(self, name: str, fields: List[str], rows: List[Dict]) -> None:
        if not rows:
            return
        with open(self.path(name), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def close(self) -> None:
        sys.stdout = self._stdout
        self._log_file.close()
```

---

## Appendix B. Exact output examples

Các file dưới đây là ví dụ schema. Số loss là placeholder, không phải kết quả thật.

### B.1. `metrics.csv`

```csv
round,mean_eval_loss,comm_cost,num_b_uploaders
1,3.0123,264,10
2,2.8841,264,10
3,2.8014,264,10
...
30,2.1029,264,10
```

Yêu cầu:

- `round` bắt đầu từ 1.
- Có đúng 30 dòng dữ liệu nếu `num_rounds = 30`.
- `comm_cost` là cost của round đó, không phải cumulative.
- `num_b_uploaders = 10` cho FlexLoRA full upload.

### B.2. `per_client_eval.csv`

```csv
round,client_id,eval_loss
1,0,2.9101
1,1,3.0442
1,2,3.2017
...
30,9,2.4431
```

Yêu cầu:

- Với 10 clients x 30 rounds phải có 300 dòng dữ liệu.
- Mỗi round phải có đủ client id `0..9`.

### B.3. `summary.json`

```json
{
  "method": "flexlora",
  "num_clients": 10,
  "num_rounds": 30,
  "final_eval_loss": 2.1029,
  "total_comm_cost": 7920.0,
  "best_eval_loss": 2.0981,
  "final_round": 30
}
```

### B.4. `config.json`

```json
{
  "method": "flexlora",
  "baseline_repo": "alibaba/FederatedScope FlexLoRA branch",
  "git_commit": "<fill-me>",
  "client_model_name": "Qwen/Qwen3-0.6B-Base",
  "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "lora_dropout": 0.05,
  "rank_pool": [4, 8, 16, 32],
  "client_ranks": {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32,
    "4": 4,
    "5": 8,
    "6": 16,
    "7": 32,
    "8": 4,
    "9": 8
  },
  "num_clients": 10,
  "num_rounds": 30,
  "local_epochs": 2,
  "local_batch_size": 8,
  "local_lr": 0.0002,
  "max_seq_len": 1024,
  "data_path": "./data/fed_wildchat.json",
  "eval_fraction": 0.1,
  "seed": 42,
  "communication_cost_unit": "rank",
  "communication_cost_formula": "sum_i r_i + sum_selected_i r_i",
  "num_b_uploaders_policy": "all_clients_every_round",
  "notes": "Separate FlexLoRA codebase adapted to FALCON protocol."
}
```

---

## Appendix C. Acceptance tests cho coding agent ở repo FlexLoRA

Coding agent bên repo FlexLoRA phải làm xong các test/check sau trước khi xem là đạt.

### C.1. Dataset split determinism test

Tạo test hoặc script kiểm tra:

```python
clients_a = load_client_datasets("./data/fed_wildchat.json", 10, 0.1, 42)
clients_b = load_client_datasets("./data/fed_wildchat.json", 10, 0.1, 42)
assert clients_a == clients_b
```

Kiểm tra thêm:

```python
assert set(clients_a.keys()) == set(range(10))
for cid in range(10):
    assert "train" in clients_a[cid]
    assert "eval" in clients_a[cid]
    assert len(clients_a[cid]["train"]) > 0
    assert len(clients_a[cid]["eval"]) > 0
```

### C.2. Rank assignment test

```python
assert assign_ranks(10, [4, 8, 16, 32]) == {
    0: 4,
    1: 8,
    2: 16,
    3: 32,
    4: 4,
    5: 8,
    6: 16,
    7: 32,
    8: 4,
    9: 8,
}
```

### C.3. Communication cost test

```python
ranks = {0: 4, 1: 8, 2: 16, 3: 32, 4: 4, 5: 8, 6: 16, 7: 32, 8: 4, 9: 8}
comm_per_round = sum(ranks.values()) + sum(ranks.values())
assert comm_per_round == 264
assert comm_per_round * 30 == 7920
```

### C.4. Output schema test

Sau một smoke run ngắn, ví dụ `num_rounds = 1`, kiểm tra:

```text
output/run_*/run.log exists
output/run_*/config.json exists
output/run_*/metrics.csv exists
output/run_*/per_client_eval.csv exists
output/run_*/summary.json exists
```

Với `num_clients = 10`, `num_rounds = 1`:

```text
metrics.csv has 1 data row
per_client_eval.csv has 10 data rows
summary.json has final_eval_loss and total_comm_cost
```

### C.5. FlexLoRA algorithm sanity test

Với fake NumPy matrices:

```python
import numpy as np

ranks = [4, 8, 16]
d, k = 32, 64
clients = []
for r in ranks:
    A = np.random.randn(r, k)
    B = np.random.randn(d, r)
    clients.append((A, B))

delta = sum(B @ A for A, B in clients) / len(clients)
U, S, Vt = np.linalg.svd(delta, full_matrices=False)
for r in ranks:
    sqrt_s = np.sqrt(S[:r])
    A_new = sqrt_s[:, None] * Vt[:r, :]
    B_new = U[:, :r] * sqrt_s[None, :]
    assert A_new.shape == (r, k)
    assert B_new.shape == (d, r)
```

Mục tiêu: đảm bảo implementation không bị sai shape khi heterogeneous ranks.

---

## Appendix D. Prompt mẫu đưa cho coding agent ở repo FlexLoRA

Khi mở repo FlexLoRA riêng, có thể đưa nguyên prompt sau:

```text
Bạn đang ở repo FlexLoRA. Hãy sửa repo này để chạy baseline FlexLoRA có thể so sánh công bằng với FALCON.

Yêu cầu quan trọng:
- Không sửa thuật toán FALCON.
- Giữ thuật toán FlexLoRA: mọi client upload A+B, server aggregate full update B@A theo sample-weighted average, SVD, redistribute theo rank client.
- Match protocol FALCON theo file guideline_baseline.md:
  - dataset ./data/fed_wildchat.json
  - split client giống FALCON
  - model Qwen/Qwen3-0.6B-Base
  - target modules q_proj,k_proj,v_proj,o_proj
  - rank_pool [4,8,16,32]
  - 10 clients, 30 rounds
  - local_epochs 2, batch_size 8, lr 2e-4, max_seq_len 1024
  - eval loss trên per-client eval split
  - communication cost theo rank-unit: sum_i r_i + sum_i r_i = 264 mỗi round
  - output files: run.log, config.json, metrics.csv, per_client_eval.csv, summary.json

Trước khi code, hãy đọc guideline_baseline.md đầy đủ.
Sau khi code, chạy smoke test 1 round nếu tài nguyên cho phép.
Nếu có điểm nào không thể match exact FALCON protocol, ghi rõ vào config.json và báo lại.
```

---

## Appendix E. Những lỗi dễ làm sai

1. **Dùng nhầm repo FLoRA/Subspace-Tuning** thay vì FlexLoRA chính thức.
2. **Dùng split dataset của repo FlexLoRA** thay vì split FALCON.
3. **Dùng chat template của Qwen** trong khi FALCON chỉ dùng plain prompt template.
4. **Dùng response-only loss** trong khi FALCON dùng full sequence labels.
5. **Dùng rank distribution của FlexLoRA paper** thay vì `[4,8,16,32,4,8,16,32,4,8]`.
6. **Tính communication bằng bytes thật** rồi so trực tiếp với FALCON rank-unit cost.
7. **Dùng optimizer/local steps khác** mà không ghi lại.
8. **Dùng metric benchmark ngoài** thay vì per-client eval loss.
9. **Không lưu per-client eval**, làm mất khả năng phân tích personalization/fairness.
10. **Dùng dynamic rank score của FALCON** trong FlexLoRA baseline.

---

## Appendix F. Minimum deliverables từ repo FlexLoRA

Khi coding agent bên repo FlexLoRA hoàn thành, phải trả lại ít nhất:

```text
1. Danh sách file đã sửa/thêm.
2. Cách chạy command chính xác.
3. Output folder path.
4. Nội dung summary.json.
5. 5 dòng đầu và 5 dòng cuối của metrics.csv.
6. Xác nhận per_client_eval.csv có đúng num_clients * num_rounds rows.
7. Xác nhận communication cost mỗi round là 264 nếu rank cố định như guideline.
8. Các mismatch nếu có so với FALCON protocol.
```

Nếu thiếu một trong các mục trên, chưa nên dùng kết quả làm main paper result.
