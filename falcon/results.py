"""Run output: tee console logs and persist metrics for later analysis.

Each run creates a timestamped directory under the configured output folder,
containing the captured console log, the run config, per-round metrics,
per-client evaluation losses, and a final summary.
"""

import csv
import dataclasses
import json
import os
import sys
from datetime import datetime
from typing import Dict, List


class _Tee:
    """Duplicate a text stream to several writers (e.g. console + log file)."""

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
        self._log_file = open(
            self.path("run.log"), "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log_file)
        print(f"[output] writing run artifacts to {self.run_dir}")

    def path(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def dump_config(self, config) -> None:
        with open(self.path("config.json"), "w", encoding="utf-8") as handle:
            json.dump(dataclasses.asdict(config), handle, indent=2)

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
