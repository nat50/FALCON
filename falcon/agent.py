"""Server-side selection agent: decide which clients upload B this round.

The agent is always an LLMAgent backed by a Gemma GGUF model (text-only). If the model cannot be loaded or its output cannot be parsed,
an error is raised.

A "client stat" is a dict: {"client_id": int, "align": float, "num_examples": int, "cost": float}.
"""

import json
import re
from typing import Dict, List

from .config import Config

from llama_cpp import Llama

class LLMAgent:

    def __init__(self, config: Config):
        
        self.llm = Llama.from_pretrained(
            repo_id=config.agent_repo_id,
            filename=config.agent_gguf_filename,
            n_ctx=config.agent_n_ctx,
            verbose=False,
        )
        print("[agent] Agent loaded.")

    def _build_prompt(self, stats: List[Dict], budget: float) -> str:
        lines = [
            "You are a federated-learning server agent.",
            "Each client trained a LoRA adapter. 'align' in [0,1] is how much the",
            "client's knowledge is COMMON (high) vs PRIVATE (low). Requesting a",
            "client's B matrix costs 'cost' units; the total budget is",
            f"{budget:.0f} units. Pick clients whose B best improves the SHARED model",
            "(prefer high align and many examples) without exceeding the budget.",
            "",
            "Clients:",
        ]
        for s in stats:
            lines.append(
                f"- id={s['client_id']} align={s['align']:.3f} "
                f"examples={s['num_examples']} cost={s['cost']:.0f}"
            )
        lines.append("")
        lines.append('Answer with ONLY a JSON list of client ids, e.g. [0, 2, 3].')
        return "\n".join(lines)

    def _parse_ids(self, text: str, valid_ids: List[int]) -> List[int]:
        match = re.search(r"\[[^\]]*\]", text)
        if not match:
            raise ValueError("no JSON list in model output")
        raw = json.loads(match.group(0))
        return [int(x) for x in raw if int(x) in valid_ids]

    def select(self, stats: List[Dict], budget: float) -> List[int]:
        prompt = self._build_prompt(stats, budget)
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        text = out["choices"][0]["message"]["content"]
        valid_ids = [s["client_id"] for s in stats]
        chosen = self._parse_ids(text, valid_ids)
        chosen = _enforce_budget(chosen, stats, budget)
        print(f"[agent] selected B-uploaders: {sorted(chosen)}")
        return chosen


def _enforce_budget(chosen: List[int], stats: List[Dict], budget: float) -> List[int]:
    """Drop the cheapest-value clients until the choice fits the budget."""
    cost_by_id = {s["client_id"]: max(s["cost"], 1.0) for s in stats}
    align_by_id = {s["client_id"]: s["align"] for s in stats}
    chosen = sorted(chosen, key=lambda cid: align_by_id.get(cid, 0.0), reverse=True)

    kept, spent = [], 0.0
    for cid in chosen:
        cost = cost_by_id.get(cid, 1.0)
        if spent + cost <= budget:
            kept.append(cid)
            spent += cost
    return kept


def make_agent(config: Config):
    return LLMAgent(config)
