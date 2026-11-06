#!/usr/bin/env python
"""Keyboard teleoperation in the Genesis viewer (v1: interaction, no recording).

Opens the real task scene (Franka + table + objects + fingertip sensors) in the
Genesis viewer and flies the TCP / gripper with the keyboard. The action stream
is the same 7-dim LIBERO-style delta the scripted expert and the policy use:
``a[0:3]`` world-frame TCP delta, ``a[3:6]`` world-frame rotvec delta, ``a[6]``
gripper (-1 open .. +1 closed). The viewer paces the simulation to wall-clock
time (``realtime_factor=1.0``), so a held key moves the arm at a stable speed.

Run from the repo root in the tempvla env:
    python scripts/teleop.py --config sim/configs/thermal_recognition/pick_cold_can.yaml

Keys:
    Left/Right     TCP -x / +x (screen left/right)
    Up/Down        TCP +y / -y (screen away/toward the camera)
    PageUp/Down    TCP +z / -z
    J / K          yaw - / + (world z)
    Shift (hold)   slow mode (0.25x)
    Space          toggle gripper open/closed
    Backspace      reset episode (re-sample object poses and temperatures)
    Q              quit

The mouse still orbits the camera (pyrender trackball). This script needs a
display (WSLg or X11); it cannot run on a headless box. If the window fails to
open on WSL2, try ``LIBGL_ALWAYS_SOFTWARE=1`` (llvmpipe, slower).

Recording is intentionally not wired here: this version is for validating the
interactive stack. The next step is a ``TeleopExpert`` behind
``EpisodeRunner.run_episode`` plus an ``episode_done`` key, which reuses the
existing HDF5 / event / burst recording unchanged.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import load_config  # noqa: E402


# --------------------------------------------------------------------- keys


def _key_codes() -> dict[str, int]:
    """Logical key name -> pyglet/Genesis key code (the Key enum is stable)."""
    from genesis.vis.keybindings import Key

    names = (
        "LEFT",
        "RIGHT",
        "UP",
        "DOWN",
        "PAGEUP",
        "PAGEDOWN",
        "J",
        "K",
        "LSHIFT",
        "RSHIFT",
        "SPACE",
        "BACKSPACE",
        "Q",
    )
    return {name: int(getattr(Key, name)) for name in names}


def _attach_key_plugin(viewer, codes: dict[str, int]):
    """Attach a ViewerPlugin that owns the teleop keys and keeps a key set.

    pyglet dispatches this plugin's handlers on the viewer thread (the viewer
    runs in a background thread on Linux). The stepping loop runs on the main
    thread, so the key set is guarded by a lock and read as a snapshot. Keys not
    in ``owned`` are passed through, leaving the default viewer keybinds
    (R/S/Z/A/H/...) working.
    """
    from genesis.vis.viewer_plugins import ViewerPlugin

    owned = set(codes.values())

    class TeleopKeys(ViewerPlugin):
        def __init__(self):
            super().__init__()
            self._lock = threading.Lock()
            self._pressed: set[int] = set()
            self._last_toggle: dict[int, float] = {}
            self._quit = False
            self._reset = False
            self._grip_toggle = False

        def _debounced(self, symbol: int, min_interval: float = 0.30) -> bool:
            """One toggle per ``min_interval``: X11/WSLg key auto-repeat sends
            repeated press events while a key is held, which would otherwise flip
            the gripper or reset the episode many times per second."""
            now = time.monotonic()
            if now - self._last_toggle.get(symbol, -1e9) < min_interval:
                return False
            self._last_toggle[symbol] = now
            return True

        def on_key_press(self, symbol, modifiers):
            symbol = int(symbol)
            if symbol not in owned:
                return None
            with self._lock:
                self._pressed.add(symbol)
                if symbol == codes["Q"]:
                    self._quit = True
                elif symbol == codes["BACKSPACE"] and self._debounced(symbol):
                    self._reset = True
                elif symbol == codes["SPACE"] and self._debounced(symbol):
                    self._grip_toggle = True
            return True

        def on_key_release(self, symbol, modifiers):
            symbol = int(symbol)
            if symbol not in owned:
                return None
            with self._lock:
                self._pressed.discard(symbol)
            return True

        def on_close(self):
            with self._lock:
                self._quit = True

        def snapshot(self) -> set[int]:
            with self._lock:
                return set(self._pressed)

        def take_quit(self) -> bool:
            with self._lock:
                value, self._quit = self._quit, False
                return value

        def take_reset(self) -> bool:
            with self._lock:
                value, self._reset = self._reset, False
                return value

        def take_grip_toggle(self) -> bool:
            with self._lock:
                value, self._grip_toggle = self._grip_toggle, False
                return value

    plugin = TeleopKeys()
    viewer.add_plugin(plugin)
    return plugin


def _action_from_keys(codes: dict[str, int], pressed: set[int], closed: bool) -> np.ndarray:
    """7-dim delta action from the currently held keys (world-frame)."""

    def held(name: str) -> float:
        return 1.0 if codes[name] in pressed else 0.0

    slow = 0.25 if (held("LSHIFT") or held("RSHIFT")) else 1.0
    trans = 0.40 * slow  # * pos_scale_m (0.05 m) -> 2 cm per policy step at 1.0
    rot = 0.10 * slow  # * rot_scale_rad (0.5 rad) -> ~2.9 deg per policy step
    return np.array(
        [
            (held("RIGHT") - held("LEFT")) * trans,
            (held("UP") - held("DOWN")) * trans,
            (held("PAGEUP") - held("PAGEDOWN")) * trans,
            0.0,
            0.0,
            (held("K") - held("J")) * rot,
            1.0 if closed else -1.0,
        ],
        dtype=np.float64,
    )


# -------------------------------------------------------------------- status


def _print_status(bundle, controller, step: int, cfg: dict, read_temperature) -> None:
    from sim.expert.actions import tcp_pose

    pos, _ = tcp_pose(bundle)
    temps = read_temperature(bundle.temperatures)
    tips = []
    for handle in bundle.temperatures:
        if handle.kind == "finger":
            value = temps[handle.name].detach().cpu().numpy().reshape(-1)[0]
            tips.append(float(value))
    tip_text = "/".join(f"{v:5.1f}" for v in tips)
    hz = float(cfg["controller"]["control_hz"])
    print(
        f"\r[{step / hz:6.1f}s] "
        f"tcp=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"grip={controller.finger_target:.3f} tip={tip_text} C",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="task YAML (default: pick_cold_can.yaml)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="episode RNG seed (default: cfg.seed; scene seed stays cfg.seed + 1000)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="do not build the task cameras (this script does not render them)",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer realtime factor (1.0 = wall clock; lower is slower)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_cameras:
        cfg["cameras"]["enabled"] = False

    from sim.expert.actions import DeltaPoseController
    from sim.scene import build_scene, sample_episode
    from sim.sensors.temperature import SENSOR_TIME_CONSTANT, read_temperature

    print("building scene; the viewer window should open...")
    bundle = build_scene(
        cfg,
        seed=int(cfg["seed"]) + 1000,
        show_viewer=True,
    )
    viewer = bundle.scene.viewer
    if viewer is None:
        print("ERROR: the scene has no viewer (build_scene(show_viewer=True) failed?)")
        return 2
    viewer.realtime_factor = float(args.realtime)

    # Frame the table with the same view the task cameras use.
    cams = cfg["cameras"]
    with viewer.lock:
        viewer.set_camera_pose(
            pos=tuple(float(v) for v in cams["agentview_pos"]),
            lookat=tuple(float(v) for v in cams["agentview_lookat"]),
        )

    codes = _key_codes()
    plugin = _attach_key_plugin(viewer, codes)

    controller = DeltaPoseController(bundle)
    rng = np.random.default_rng(
        int(args.seed) if args.seed is not None else int(cfg["seed"])
    )
    sim_dt = float(cfg["controller"]["sim_dt"])
    decimation = int(cfg["controller"]["decimation"])
    status_every = max(1, int(round(float(cfg["controller"]["control_hz"]))))

    def step_sim(count: int = 1) -> bool:
        """Step the sim; False means the viewer window was closed.

        ``scene.step()`` drives the viewer, and Genesis raises
        ``GenesisException("Viewer closed.")`` on the next step after the window
        disappears. That is a normal exit here; any other error is re-raised (and
        ``is_alive()`` itself re-raises a viewer-thread crash).
        """
        try:
            for _ in range(count):
                bundle.step()
        except Exception:
            if viewer.is_alive():
                raise
            return False
        return True

    def reset_episode() -> bool:
        ep = sample_episode(bundle.cfg, rng)
        bundle.reset_episode(ep)
        # One step publishes the post-reset link poses (the runner does the
        # same), then settle the fingertip RC filter, which the engine zeroes on
        # reset, so the HUD starts at ambient instead of 0 C.
        if not step_sim(1):
            return False
        tau = float(SENSOR_TIME_CONSTANT)
        if tau > 0.0 and bundle.temperatures:
            if not step_sim(int(np.ceil(5.0 * tau / sim_dt))):
                return False
        controller.reset()
        return True

    if not reset_episode():
        print("viewer closed before the first step")
        return 0
    step = 0
    grip_closed = False
    print(
        "teleop ready: click the window once for keyboard focus, then\n"
        "  arrows move in x/y, PageUp/PageDown in z, J/K yaw, "
        "hold Shift for 0.25x speed\n"
        "  Space toggles the gripper, Backspace re-samples the episode, Q quits"
    )
    try:
        while True:
            if plugin.take_quit() or not viewer.is_alive():
                break
            if plugin.take_reset():
                if not reset_episode():
                    break
                step = 0
                grip_closed = False
                print("\n[reset]")
                continue
            if plugin.take_grip_toggle():
                grip_closed = not grip_closed

            action = _action_from_keys(codes, plugin.snapshot(), grip_closed)
            controller.step(action)
            if not step_sim(decimation):
                break
            step += 1
            if step % status_every == 0:
                _print_status(bundle, controller, step, cfg, read_temperature)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            viewer.stop()
        except Exception:  # noqa: BLE001 - closing a dead viewer is harmless
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
