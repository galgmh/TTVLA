"""Shared helpers for the remaining scripts (config loading, timing, memory).

Kept dependency-free (stdlib only) so a script can report a broken engine
rather than crashing on its own import.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def meminfo() -> dict[str, float]:
    """Selected /proc/meminfo entries in GiB (empty dict off-Linux)."""
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key.strip()] = float(rest.strip().split()[0]) / 1024**2
    except OSError:
        pass
    return out


class Timer:
    """With-block wall-clock timer: ``with Timer() as t: ...; t.s``."""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.s = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.s = time.perf_counter() - self._t0

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.s:.2f}s"


def load_config(path: str | None, overrides: dict[str, Any] | None = None):
    """Task config dict loaded from a YAML file.

    With no explicit path, the repo's single task config is used; there is no
    built-in default because the YAML is the only source of task values.
    """
    from sim.config import load_task_config

    if path is None:
        default = (
            Path(__file__).resolve().parents[1]
            / "sim" / "configs" / "thermal_recognition" / "pick_cold_can.yaml"
        )
        if default.is_file():
            path = str(default)
        else:
            raise FileNotFoundError(
                "no --config and sim/configs/thermal_recognition/pick_cold_can.yaml "
                "is missing; the task config must come from a YAML file"
            )
    print(f"  config: {path}")
    cfg = load_task_config(path)
    for key, value in (overrides or {}).items():
        cfg[key] = value
    return cfg
