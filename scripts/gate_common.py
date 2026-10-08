"""Shared helpers for the gate scripts (gate0/gate1/gate2).

Kept dependency-free (stdlib + numpy) so a gate can report a broken engine
rather than crashing on its own import.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class Gate:
    """Collects pass/fail/warn lines and prints them in a stable format."""

    def __init__(self, title: str = ""):
        self.title = title
        self.failures: list[str] = []
        self.warnings: list[str] = []
        if title:
            print(f"########## {title} ##########")

    def ok(self, name: str, detail: str = "") -> None:
        print(f"  [ ok ] {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = "") -> None:
        self.failures.append(name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        self.warnings.append(name)
        print(f"  [warn] {name}" + (f" — {detail}" if detail else ""))

    def info(self, text: str = "") -> None:
        print(f"  {text}" if text else "")

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(name, detail)
        return bool(condition)

    def summary(self, next_step: str = "") -> int:
        print(f"\n=== summary ===")
        print(f"failures: {len(self.failures)}  warnings: {len(self.warnings)}")
        for f in self.failures:
            print(f"  FAIL {f}")
        if not self.failures and next_step:
            print(next_step)
        return 1 if self.failures else 0


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
    """TaskConfig from a YAML file, or the dataclass defaults.

    With no explicit path, configs/pick_hot_cylinder.yaml is used when present
    (the dataclass defaults alone have no material thermal properties).
    """
    from tempvla.config import TaskConfig, load_task_config

    if path is None:
        default = Path(__file__).resolve().parents[1] / "configs" / "pick_hot_cylinder.yaml"
        if default.is_file():
            path = str(default)
        else:
            print("  [warn] no --config and configs/pick_hot_cylinder.yaml is missing; using dataclass defaults")
    if path:
        print(f"  config: {path}")
    cfg = load_task_config(path) if path else TaskConfig()
    for key, value in (overrides or {}).items():
        setattr(cfg, key, value)
    return cfg


def fmt_floats(values, precision: int = 4) -> str:
    import numpy as np

    return str(np.round(np.asarray(values, dtype=float), precision).tolist())
