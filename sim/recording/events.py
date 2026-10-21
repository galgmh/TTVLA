"""Contact-event detection (L3) and burst-window capture (L4).

Pure logic on numpy payloads — no Genesis import — so it is unit-testable and
reusable by both the live runner and offline tools.

L3 (contact events): one row per rising edge of finger-object contact, with the
   peak force / depth observed while the contact stayed active.
L4 (burst windows): a ring buffer keeps the last ``pre_steps`` sim-substep
   frames; a trigger snapshots it and keeps appending until ``post_steps``
   frames have elapsed, giving 200 Hz windows around each contact onset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class ContactEvent:
    event_id: int
    start_step: int  # policy step index where the rising edge was seen
    start_frame: int  # sim-substep index
    time_s: float
    link_names: tuple[str, ...]  # links in contact at the rising edge
    peak_force_n: float = 0.0
    max_depth_m: float = 0.0
    end_step: int | None = None  # last policy step with contact still active
    end_time_s: float | None = None


class ContactMonitor:
    """Tracks per-link contact booleans at policy rate; emits L3 events."""

    def __init__(self, link_names: Sequence[str], *, force_threshold_n: float = 0.0):
        self.link_names = tuple(link_names)
        self.force_threshold_n = float(force_threshold_n)
        self.events: list[ContactEvent] = []
        self._active: ContactEvent | None = None
        self._prev_bool = np.zeros(len(self.link_names), dtype=bool)

    def update(
        self,
        *,
        step: int,
        frame: int,
        time_s: float,
        link_bool: np.ndarray | None,
        link_force: np.ndarray | None,
        depth: Mapping[str, np.ndarray] | None = None,
        force_grid: Mapping[str, np.ndarray] | None = None,
    ) -> ContactEvent | None:
        """Feed one policy step. Returns the event on a rising edge (else None)."""
        if link_bool is None:
            return None
        current = np.asarray(link_bool).reshape(-1)[: len(self.link_names)].astype(bool)
        if self.force_threshold_n > 0 and link_force is not None:
            mag = np.linalg.norm(np.asarray(link_force).reshape(-1, 3)[: len(self.link_names)], axis=-1)
            current &= mag > self.force_threshold_n

        peak_force = self._peak_force(link_force)
        max_depth = self._max_depth(depth, force_grid)

        rising = current & ~self._prev_bool
        emitted: ContactEvent | None = None
        if rising.any():
            ev = ContactEvent(
                event_id=len(self.events),
                start_step=int(step),
                start_frame=int(frame),
                time_s=float(time_s),
                link_names=tuple(self.link_names[i] for i in np.nonzero(rising)[0]),
                peak_force_n=peak_force,
                max_depth_m=max_depth,
            )
            self.events.append(ev)
            self._active = ev
            emitted = ev

        if self._active is not None:
            if current.any():
                self._active.peak_force_n = max(self._active.peak_force_n, peak_force)
                self._active.max_depth_m = max(self._active.max_depth_m, max_depth)
                self._active.end_step = int(step)
                self._active.end_time_s = float(time_s)
            elif not rising.any():
                self._active = None

        self._prev_bool = current
        return emitted

    def _peak_force(self, link_force: np.ndarray | None) -> float:
        if link_force is None:
            return 0.0
        mag = np.linalg.norm(np.asarray(link_force).reshape(-1, 3)[: len(self.link_names)], axis=-1)
        return float(mag.max(initial=0.0))

    @staticmethod
    def _max_depth(depth: Mapping[str, np.ndarray] | None, force_grid: Mapping[str, np.ndarray] | None) -> float:
        src = depth if depth else force_grid
        if not src:
            return 0.0
        return float(max((np.asarray(v).max(initial=0.0) for v in src.values()), default=0.0))

    # ------------------------------------------------------------------ export

    def table(self, num_links: int | None = None) -> dict[str, np.ndarray]:
        """Columnar event table for HDF5 (events/contact_events/)."""
        n = len(self.events)
        links = self.link_names if num_links is None else self.link_names[:num_links]
        link_bool = np.zeros((n, len(links)), dtype=bool)
        for e_i, ev in enumerate(self.events):
            for name in ev.link_names:
                if name in links:
                    link_bool[e_i, links.index(name)] = True
        return {
            "event_id": np.arange(n, dtype=np.int32),
            "start_step": np.array([e.start_step for e in self.events], dtype=np.int32),
            "start_frame": np.array([e.start_frame for e in self.events], dtype=np.int32),
            "time_s": np.array([e.time_s for e in self.events], dtype=np.float32),
            "end_step": np.array([e.end_step if e.end_step is not None else -1 for e in self.events], dtype=np.int32),
            "peak_force_n": np.array([e.peak_force_n for e in self.events], dtype=np.float32),
            "max_depth_m": np.array([e.max_depth_m for e in self.events], dtype=np.float32),
            "link_bool": link_bool,
        }


@dataclass
class BurstWindow:
    trigger_frame: int
    start_frame: int
    end_frame: int  # inclusive; set when closed
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)
    closed: bool = False

    def append(self, payload: Mapping[str, np.ndarray]) -> None:
        for key, value in payload.items():
            self.frames.setdefault(key, []).append(np.asarray(value))


class BurstRecorder:
    """Ring buffer + open windows of sim-substep frames (L4, 200 Hz)."""

    def __init__(self, *, pre_steps: int, post_steps: int, max_windows: int):
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)
        self.max_windows = int(max_windows)
        self._ring: deque[tuple[int, dict[str, np.ndarray]]] = deque(maxlen=self.pre_steps + 1)
        self._open: list[BurstWindow] = []
        self.windows: list[BurstWindow] = []
        self.dropped = 0

    def push(self, frame: int, payload: Mapping[str, np.ndarray]) -> None:
        """Record one sim-substep frame; call every substep while enabled."""
        snapshot = {k: np.asarray(v).copy() for k, v in payload.items()}
        self._ring.append((int(frame), snapshot))
        for win in self._open:
            win.append(snapshot)
            if frame >= win.end_frame:
                win.closed = True
        self._open = [w for w in self._open if not w.closed]

    def trigger(self, frame: int) -> bool:
        """Open a window around a contact onset. False if capacity exceeded."""
        if len(self.windows) >= self.max_windows:
            self.dropped += 1
            return False
        win = BurstWindow(trigger_frame=int(frame), start_frame=int(frame) - self.pre_steps, end_frame=int(frame) + self.post_steps)
        trigger_key = None
        for f, payload in self._ring:
            win.append(payload)
            if trigger_key is None:
                trigger_key = next(iter(payload), None)
            if f >= frame:
                break
        # Pad with the newest frame if the ring was shorter than pre_steps.
        while trigger_key is not None and len(win.frames[trigger_key]) < self.pre_steps + 1:
            win.append(self._ring[-1][1])
        self._open.append(win)
        self.windows.append(win)
        return True

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    def flush(self) -> None:
        """Close windows still open at episode end."""
        for win in self._open:
            win.closed = True
        self._open = []

    def stack(self) -> list[dict[str, np.ndarray]]:
        """One dict of (T, *shape) arrays per window."""
        out = []
        for win in self.windows:
            out.append({k: np.stack(v, axis=0) for k, v in win.frames.items() if v})
        return out

    def metadata(self) -> dict[str, np.ndarray]:
        return {
            "trigger_frame": np.array([w.trigger_frame for w in self.windows], dtype=np.int32),
            "start_frame": np.array([w.start_frame for w in self.windows], dtype=np.int32),
            "end_frame": np.array([w.end_frame for w in self.windows], dtype=np.int32),
            "num_frames": np.array([len(next(iter(w.frames.values()), [])) for w in self.windows], dtype=np.int32),
        }
