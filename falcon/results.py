"""Persist experiment histories and compact summaries to disk."""

import csv
import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _slug(value: str) -> str:
    """Make a filesystem-safe label while keeping it human-readable."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _series_to_rows(series: Iterable) -> List[Dict[str, float]]:
    return [{"round": int(round_id), "value": float(value)}
            for round_id, value in series]


def _metric_dict(metrics: Dict[str, Iterable]) -> Dict[str, List[Dict[str, float]]]:
    return {name: _series_to_rows(series) for name, series in metrics.items()}


def history_to_dict(history) -> Dict:
    """Convert a Flower History object into JSON-serializable data."""
    return {
        "losses_distributed": _series_to_rows(
            getattr(history, "losses_distributed", [])
        ),
        "losses_centralized": _series_to_rows(
            getattr(history, "losses_centralized", [])
        ),
        "metrics_distributed_fit": _metric_dict(
            getattr(history, "metrics_distributed_fit", {})
        ),
        "metrics_distributed_evaluate": _metric_dict(
            getattr(history, "metrics_distributed", {})
        ),
        "metrics_centralized": _metric_dict(
            getattr(history, "metrics_centralized", {})
        ),
    }


def summarize_history(history, config) -> Dict:
    """Return the compact metrics table row for one run."""
    losses = getattr(history, "losses_distributed", [])
    final_loss = float(losses[-1][1]) if losses else None
    comm_series = getattr(history, "metrics_distributed_fit", {}).get("comm_cost", [])
    b_series = getattr(history, "metrics_distributed_fit", {}).get(
        "num_b_uploaders", []
    )

    total_comm = float(sum(value for _, value in comm_series))
    total_b_uploads = float(sum(value for _, value in b_series))
    possible_b_uploads = float(config.num_clients * config.num_rounds)
    b_upload_fraction = (
        total_b_uploads / possible_b_uploads if possible_b_uploads else None
    )

    return {
        "method": config.selection_mode,
        "dataset": config.dataset_name,
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "max_train_per_client": config.max_train_per_client,
        "max_test_per_client": config.max_test_per_client,
        "final_eval_loss": final_loss,
        "final_perplexity": math.exp(final_loss) if final_loss is not None else None,
        "total_comm_cost": total_comm,
        "total_b_uploads": total_b_uploads,
        "b_upload_fraction": b_upload_fraction,
    }


def _config_to_dict(config) -> Dict:
    return asdict(config) if is_dataclass(config) else dict(vars(config))


def _client_summary(client_data: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for client_id, split in sorted(client_data.items()):
        rows.append({
            "client_id": int(client_id),
            "category": split.get("category", ""),
            "num_train": len(split.get("train", [])),
            "num_test": len(split.get("test", [])),
        })
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_run_results(
    history,
    config,
    client_data: Dict[int, Dict],
    client_ranks: Dict[int, int],
    output_root: str = "outputs/results",
    run_id: Optional[str] = None,
) -> Path:
    """Save one simulation result and return its output directory."""
    run_id = run_id or _timestamp()
    output_dir = (
        Path(output_root)
        / _slug(config.dataset_name)
        / _slug(config.selection_mode)
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_history(history, config)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "config": _config_to_dict(config),
        "client_ranks": {str(k): int(v) for k, v in sorted(client_ranks.items())},
        "clients": _client_summary(client_data),
        "history": history_to_dict(history),
    }

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(output_dir / "summary.csv", [summary])
    _write_csv(output_dir / "clients.csv", payload["clients"])
    print(f"[results] saved run results to {output_dir}")
    return output_dir


def save_comparison(
    rows: List[Dict],
    config,
    output_root: str = "outputs/results",
    run_id: Optional[str] = None,
) -> Path:
    """Save the multi-method comparison table from run_experiments.py."""
    run_id = run_id or _timestamp()
    output_dir = Path(output_root) / _slug(config.dataset_name) / "comparison" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": config.dataset_name,
            "rows": rows,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[results] saved comparison results to {output_dir}")
    return output_dir
