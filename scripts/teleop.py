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

Keys are defined in the ARM BASE FRAME (``robot.base_pos`` / ``robot.base_quat``
in the YAML; in this task the base sits at the world origin with identity
orientation, so that frame coincides with the world frame):

    x -> toward the table, y -> the arm's left, z -> up   (right-handed)

    Up/Down        TCP +x / -x (toward / away from the table)
    Left/Right     TCP +y / -y (the arm's left / right)
    PageUp/Down    TCP +z / -z (up / down)
    Y / U          rotate about the arm x axis - / +
    N / M          rotate about the arm y axis - / +
    J / K          rotate about the arm z axis (yaw) - / +
    Shift (hold)   slow mode (0.25x)
    Space          toggle gripper open/closed
    Backspace      reset episode (re-sample object poses and temperatures)
    Q              quit

The action that leaves this script is still the world-frame 7-dim delta above
(the dataset/expert contract); the key intent is rotated into the world frame
with the base orientation before it is sent. Note the default viewer camera
looks along +y, so +x (Up) reads as "toward the image right" on screen; orbit
the view for a top-down reading, the keys do not depend on it.

The mouse still orbits the camera (pyrender trackball). This script needs a
display (WSLg or X11); it cannot run on a headless box. If the window fails to
open on WSL2, try ``LIBGL_ALWAYS_SOFTWARE=1`` (llvmpipe, slower).

The status block (once per second) doubles as a key diagnostic: it shows the
7-dim action, whether this step issued one (``cmd=move``) or re-issued the
frozen joint command (``cmd=hold``), the IK residual of the last solve, and the
last key symbol the window received (``last=``). That separates "key never
reached the window" from "action applied but no visible motion".

Camera preview: a second (OpenCV) window shows the two task cameras -- the head
camera ``agentview`` and the wrist camera ``eye_in_hand`` -- at ``--preview-hz``
(default 5). Frames come from ``SceneBundle.render_cameras()``, the same call
``ObsBuilder`` makes when the dataset records ``agentview_rgb`` /
``eye_in_hand_rgb``, so the preview shows the recorder's view. It is a second
window rather than an overlay because Genesis 1.4.1 has no in-viewer camera
panel: its ImGui overlay requires ``imgui_bundle`` (not installed here) and
does not display camera images anyway. The window itself runs in a small child
process: creating a Qt window in this process while the Genesis viewer thread
renders deadlocks inside Qt's platform-plugin init on WSLg (observed as
``cv2.namedWindow`` never returning), so frames are composited here and piped
to a child that owns the window. Preview resolution defaults to 256x256 because
every preview frame is a full raster render (llvmpipe here); ``--cam-res 0``
keeps the YAML resolution (512) and ``--cam-res 512`` is the dataset-exact
view. ``--no-preview`` turns the window off, ``--no-cameras`` skips building
the cameras entirely.

Grip strength: the fingers are position-controlled (YAML ``finger_kp`` = 100
N/m), so the squeeze force at contact is ``kp * (closed target - finger
position)`` -- not the ``finger_force_max`` limit, which is never reached. That
gives only ~1.9-2.5 N per pad at these objects (the 0.5 kg iron cube cannot be
lifted at all). ``--grip-kp`` overrides the finger stiffness for this run only
(400 -> ~8-10 N per pad at these objects); the task YAML, the scripted expert
and any recorded dataset keep their own value. kp lives in the solver's
dyn_info, which ``scene.reset()`` does not touch, so one application after
build survives every episode reset.

Recording is intentionally not wired here: this version is for validating the
interactive stack. The next step is a ``TeleopExpert`` behind
``EpisodeRunner.run_episode`` plus an ``episode_done`` key, which reuses the
existing HDF5 / event / burst recording unchanged.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
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
        "Y",
        "U",
        "N",
        "M",
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
            self._last_press: int | None = None
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
            with self._lock:
                self._last_press = symbol
                if symbol not in owned:
                    return None
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

        def on_deactivate(self):
            # pyglet dispatches FocusOut here. A key released while the window
            # is unfocused never produces an on_key_release, so without this a
            # stale movement key (e.g. PageDown) would keep commanding motion
            # and the arm would slowly sink on its own.
            with self._lock:
                self._pressed.clear()

        def clear_pressed(self) -> None:
            """Drop every held key (used after an episode reset)."""
            with self._lock:
                self._pressed.clear()

        def last_press(self) -> int | None:
            with self._lock:
                return self._last_press

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


def _action_from_keys(
    codes: dict[str, int], pressed: set[int], closed: bool, base_rot: np.ndarray
) -> np.ndarray:
    """7-dim delta action from the held keys, expressed in the ARM BASE FRAME.

    ``base_rot`` maps base-frame vectors to world (its columns are the arm
    frame's x/y/z axes in world, from ``robot.base_quat``). Keys are assigned
    in that frame: Up/Down = +-x (toward / away from the table), Left/Right =
    +-y (the arm's left / right), PageUp/PageDown = +-z, and Y/U, N/M, J/K
    rotate about x/y/z. The result is rotated into the world frame here,
    because the 7-dim action stream is world-frame by contract (expert,
    dataset, policy).
    """

    def held(name: str) -> float:
        return 1.0 if codes[name] in pressed else 0.0

    slow = 0.25 if (held("LSHIFT") or held("RSHIFT")) else 1.0
    trans = 0.40 * slow  # * pos_scale_m (0.05 m) -> 2 cm per policy step at 1.0
    rot = 0.10 * slow  # * rot_scale_rad (0.5 rad) -> ~2.9 deg per policy step
    local = np.array(
        [
            (held("UP") - held("DOWN")) * trans,  # +x: toward the table
            (held("LEFT") - held("RIGHT")) * trans,  # +y: the arm's left
            (held("PAGEUP") - held("PAGEDOWN")) * trans,  # +z: up
            (held("U") - held("Y")) * rot,  # about the arm x axis
            (held("M") - held("N")) * rot,  # about the arm y axis
            (held("K") - held("J")) * rot,  # about the arm z axis
            1.0 if closed else -1.0,
        ],
        dtype=np.float64,
    )
    action = local.copy()
    action[:3] = base_rot @ local[:3]
    action[3:6] = base_rot @ local[3:6]
    return action


# ----------------------------------------------------------------- preview


# The preview window lives in a CHILD process on purpose. Creating a Qt window
# (cv2.namedWindow) in this process while the Genesis viewer thread renders
# deadlocks inside Qt's platform-plugin init on WSLg -- observed as
# namedWindow() never returning while the viewer thread stayed healthy. A child
# process gets its own X connection and toolkit state, so the two GUIs cannot
# block each other. Frames travel over stdin: ``<II`` height, width (uint32,
# little-endian) followed by raw BGR bytes.
_DISPLAY_CHILD = r"""
import struct
import sys

import cv2
import numpy as np

title = sys.argv[1]
while True:
    head = sys.stdin.buffer.read(8)
    if len(head) < 8:
        break
    height, width = struct.unpack("<II", head)
    size = height * width * 3
    data = sys.stdin.buffer.read(size)
    if len(data) < size:
        break
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    cv2.imshow(title, image)
    cv2.waitKey(1)
    try:
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break  # user closed the preview window
    except cv2.error:
        break
cv2.destroyAllWindows()
"""


class _CameraPreview:
    """Live agentview / eye_in_hand preview in a separate OpenCV process.

    Frames come from ``SceneBundle.render_cameras()`` -- the same path the
    dataset recorder uses -- so this shows the recorder's view, not a
    viewer-side re-render. The grid is composited here (cv2 is used as a plain
    image library; no GUI in this process) and piped to a child that owns the
    window; see the comment above ``_DISPLAY_CHILD`` for why.

    Clicking the preview window takes keyboard focus away from the Genesis
    window. The key plugin clears held keys on FocusOut, so no key sticks;
    click the Genesis window again to keep flying.
    """

    def __init__(self, bundle, names: tuple[str, ...] = ("agentview", "eye_in_hand")):
        import cv2

        self._cv2 = cv2
        self._bundle = bundle
        self._names = names
        self._title = "teleop cameras: agentview | eye_in_hand"
        self._closed = False
        self._proc = None
        self._announced = False
        self._spawn_display()

    def _spawn_display(self) -> None:
        # WSLg exposes both a Wayland socket and X11, but the Qt bundled inside
        # the opencv-python wheel ships only the xcb platform plugin
        # (cv2/qt/plugins/platforms/libqxcb.so). If Qt picks Wayland, window
        # creation aborts with a platform-plugin error; pin xcb whenever an X
        # display exists (the Genesis viewer itself needs X11 anyway).
        env = dict(os.environ)
        if env.get("DISPLAY") and not env.get("QT_QPA_PLATFORM"):
            env["QT_QPA_PLATFORM"] = "xcb"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c", _DISPLAY_CHILD, self._title],
                stdin=subprocess.PIPE,
                env=env,
            )
        except OSError as exc:  # a broken preview must not kill teleop
            print(f"camera preview disabled (cannot start the display process: {exc})")
            self._closed = True
            return
        print("camera preview: window process started", flush=True)

    def update(self) -> None:
        if self._closed or self._proc is None:
            return
        cv2 = self._cv2
        frames = self._bundle.render_cameras()
        tiles = []
        for name in self._names:
            frame = frames.get(name)
            if frame is None:
                continue
            rgb = np.asarray(frame)
            if rgb.ndim == 4:  # (n_envs, H, W, 3)
                rgb = rgb[0]
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                continue
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            bgr = cv2.copyMakeBorder(bgr, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            label = name.upper()
            # Dark halo first so the label reads on bright table/object pixels.
            cv2.putText(bgr, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                bgr, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA
            )
            tiles.append(bgr)
        if not tiles:
            return
        if len(tiles) > 1:
            height = max(t.shape[0] for t in tiles)
            tiles = [
                tile
                if tile.shape[0] == height
                else cv2.resize(tile, (tile.shape[1], height), interpolation=cv2.INTER_NEAREST)
                for tile in tiles
            ]
        grid = np.ascontiguousarray(np.hstack(tiles), dtype=np.uint8)
        try:
            stdin = self._proc.stdin
            stdin.write(struct.pack("<II", *grid.shape[:2]))
            stdin.write(grid.tobytes())
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:  # preview window closed / child gone
            print(f"camera preview stopped ({exc})")
            self._closed = True
            return
        if not self._announced:
            print(f"camera preview: {grid.shape[1]}x{grid.shape[0]} frames streaming", flush=True)
            self._announced = True

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self._closed = True
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF -> the child closes the window and exits
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()


# -------------------------------------------------------------------- status


def _print_status(
    bundle,
    controller,
    step: int,
    cfg: dict,
    read_temperature,
    action: np.ndarray,
    issued: bool,
    info,
    last_symbol: int | None,
    codes: dict[str, int],
) -> None:
    from sim.expert.actions import tcp_pose
    from sim.rotations import quat_to_matrix

    pos, quat = tcp_pose(bundle)
    # Hand z axis (flange -> grip center): where the fingers point, so a tilt is
    # visible on the HUD even when the viewer is orbited.
    axis = quat_to_matrix(quat)[:, 2]
    temps = read_temperature(bundle.temperatures)
    tips = []
    for handle in bundle.temperatures:
        if handle.kind == "finger":
            value = temps[handle.name].detach().cpu().numpy().reshape(-1)[0]
            tips.append(float(value))
    tip_text = "/".join(f"{v:5.1f}" for v in tips)
    hz = float(cfg["controller"]["control_hz"])

    a = np.asarray(action, dtype=np.float64).reshape(-1)
    ik = "n/a" if info is None else f"{info.ik_pos_err_m:.1e}/{info.ik_rot_err_rad:.1e}"
    if last_symbol is None:
        key_name = "-"
    else:
        key_name = next((n for n, c in codes.items() if c == last_symbol), "")
        if not key_name:
            key_name = f"0x{last_symbol:x}"
    print(
        f"[{step / hz:6.1f}s] "
        f"tcp=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"dir=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}) "
        f"grip={controller.finger_target:.3f} tip={tip_text} C"
    )
    print(
        f"          a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f} | "
        f"{a[3]:+.2f},{a[4]:+.2f},{a[5]:+.2f}) "
        f"cmd={'move' if issued else 'hold'} ik={ik} last={key_name}"
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
        help="do not build the task cameras (also disables the live camera preview)",
    )
    parser.add_argument(
        "--cam-res",
        type=int,
        default=256,
        help="preview camera resolution in px, square (default 256; 0 = keep the "
        "YAML value, 512 = dataset-exact; this run only, nothing is recorded)",
    )
    parser.add_argument(
        "--preview-hz",
        type=float,
        default=5.0,
        help="live camera preview update rate in Hz (default 5; each update renders "
        "both cameras once, so keep it low on software GL)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="do not open the camera preview window",
    )
    parser.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="viewer realtime factor (1.0 = wall clock; lower is slower)",
    )
    parser.add_argument(
        "--grip-kp",
        type=float,
        default=0.0,
        help="finger PD stiffness in N/m per finger for THIS run only (0 = keep "
        "the YAML value, 100). The fingers are position-controlled, so the "
        "squeeze force is kp x (closed target - finger position): 100 gives "
        "~1.9-2.5 N per pad here, 400 gives ~8-10 N. Nothing is recorded.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_cameras:
        cfg["cameras"]["enabled"] = False
    elif args.cam_res > 0:
        # Preview-only override. Teleop records nothing, and every preview frame
        # is a full raster render, so 256x256 costs ~4x less than the YAML's 512.
        side = int(args.cam_res)
        cfg["cameras"]["resolution"] = [side, side]

    from sim.expert.actions import DeltaPoseController
    from sim.rotations import quat_to_matrix
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

    # Arm base frame in world: columns are x (toward the table), y (the arm's
    # left) and z (up) as the robot is mounted. Identity in this task, but read
    # from the config so the key mapping follows the robot if the base moves.
    base_rot = quat_to_matrix(np.asarray(cfg["robot"]["base_quat"], dtype=np.float64))

    if args.grip_kp > 0.0:
        # Teleop-only squeeze-force override, see the module docstring. The
        # finger PD gains live in the solver's dyn_info (act_gain/act_bias);
        # scene.reset() rewrites only the DOF state and ctrl mode, so applying
        # it once after build survives every episode reset.
        stiffness = float(args.grip_kp)
        bundle.robot.set_dofs_kp(
            np.full(len(bundle.finger_dofs), stiffness, dtype=np.float32),
            bundle.finger_dofs,
        )
        print(
            f"grip stiffness: {stiffness:g} N/m per finger "
            f"(YAML {float(cfg['robot']['finger_kp'][0]):g}); teleop-only override"
        )

    controller = DeltaPoseController(bundle)
    rng = np.random.default_rng(
        int(args.seed) if args.seed is not None else int(cfg["seed"])
    )
    sim_dt = float(cfg["controller"]["sim_dt"])
    decimation = int(cfg["controller"]["decimation"])
    control_hz = float(cfg["controller"]["control_hz"])
    status_every = max(1, int(round(control_hz)))
    preview_every = max(1, int(round(control_hz / max(float(args.preview_hz), 0.1))))

    preview = None
    if not args.no_preview:
        if not bundle.cameras:
            print("camera preview disabled: no task cameras were built")
        else:
            try:
                preview = _CameraPreview(bundle)
            except ImportError as exc:  # OpenCV absent
                print(f"camera preview disabled ({exc}); install opencv-python for it")
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
    action = np.zeros(7, dtype=np.float64)
    last_info = None
    issued = False
    if preview is not None:
        preview.update()  # show the first frames right away
    print(
        "teleop ready (arm base frame: x toward the table, y arm-left, z up)\n"
        "  Up/Down = +-x (toward/away from the table), "
        "Left/Right = +-y (arm left/right), PageUp/PageDown = +-z (up/down)\n"
        "  Y/U, N/M, J/K rotate about the arm x/y/z axes\n"
        "  hold Shift for 0.25x speed | Space grip, Backspace reset, Q quit"
    )
    if preview is not None:
        res = cfg["cameras"]["resolution"]
        print(
            f"  camera preview: agentview + eye_in_hand at {res[0]}x{res[1]}, "
            f"~{args.preview_hz:g} Hz (click back into this window to keep flying)"
        )
    try:
        while True:
            if plugin.take_quit() or not viewer.is_alive():
                break
            if plugin.take_reset():
                if not reset_episode():
                    break
                plugin.clear_pressed()
                step = 0
                grip_closed = False
                action = np.zeros(7, dtype=np.float64)
                last_info = None
                print("\n[reset]")
                continue
            grip_before = grip_closed
            if plugin.take_grip_toggle():
                grip_closed = not grip_closed

            action = _action_from_keys(codes, plugin.snapshot(), grip_closed, base_rot)
            grip_changed = grip_closed != grip_before
            # DeltaPoseController re-anchors every command on the *measured* TCP
            # (its anti-windup design, see sim/expert/actions.py). With a zero
            # action that setpoint is exactly the pose gravity has already sagged
            # the arm into, so the PD deflection would be re-anchored away and
            # accumulate: an idle arm slowly droops. A zero action therefore
            # re-issues the frozen joint command instead of re-solving IK;
            # re-anchoring resumes on the next commanded motion.
            if np.any(action[:6] != 0.0) or grip_changed:
                last_info = controller.step(action)
                issued = True
            else:
                controller.hold()
                issued = False
            if not step_sim(decimation):
                break
            step += 1
            if preview is not None and step % preview_every == 0:
                preview.update()
            if step % status_every == 0:
                _print_status(
                    bundle,
                    controller,
                    step,
                    cfg,
                    read_temperature,
                    action,
                    issued,
                    last_info,
                    plugin.last_press(),
                    codes,
                )
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if preview is not None:
            preview.close()
        try:
            viewer.stop()
        except Exception:  # noqa: BLE001 - closing a dead viewer is harmless
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
