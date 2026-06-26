"""Server-side selection agent: decide which clients upload B this round.

Two interchangeable agents:
  - HeuristicAgent: deterministic greedy knapsack on score = align * n / cost.
  - LLMAgent: prompts a Gemma GGUF model (text-only) to make the choice, and
    transparently falls back to the heuristic if llama-cpp or the model is missing
    or the model returns something unparsable.

A "client stat" is a dict: {"client_id": int, "align": float, "num_examples": int, "cost": float}.
"""

import json
import re
from typing import Dict, List

from .config import Config


def _greedy_select(stats: List[Dict], budget: float) -> List[int]:
    """Pick clients by descending benefit-per-cost until the budget is spent."""
    scored = []
    for s in stats:
        cost = max(s["cost"], 1.0)
        score = (s["align"] * s["num_examples"]) / cost
        scored.append((score, s["client_id"], cost))
    scored.sort(reverse=True)

    selected, spent = [], 0.0
    for _, client_id, cost in scored:
        if spent + cost <= budget:
            selected.append(client_id)
            spent += cost
    return selected


class HeuristicAgent:
    """Deterministic baseline selector."""

    def select(self, stats: List[Dict], budget: float) -> List[int]:
        chosen = _greedy_select(stats, budget)
        print(f"[agent:heuristic] selected B-uploaders: {sorted(chosen)}")
        return chosen


class LLMAgent:
    """Gemma-GGUF selector with a heuristic fallback."""

    def __init__(self, config: Config):
        self.fallback = HeuristicAgent()
        self.llm = None
        try:
            from llama_cpp import Llama  # imported lazily; optional dependency

            self.llm = Llama.from_pretrained(
                repo_id=config.agent_repo_id,
                filename=config.agent_gguf_filename,
                n_ctx=config.agent_n_ctx,
                verbose=False,
            )
            print("[agent:llm] Gemma GGUF loaded.")
        except Exception as error:  # noqa: BLE001 - any failure -> heuristic
            print(f"[agent:llm] unavailable ({error}); using heuristic fallback.")

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
        if self.llm is None:
            return self.fallback.select(stats, budget)
        try:
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
            print(f"[agent:llm] selected B-uploaders: {sorted(chosen)}")
            return chosen
        except Exception as error:  # noqa: BLE001 - parsing/model error -> heuristic
            print(f"[agent:llm] selection failed ({error}); using heuristic.")
            return self.fallback.select(stats, budget)


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
    if config.agent_kind == "llm":
        return LLMAgent(config)
    return HeuristicAgent()
