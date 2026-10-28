#!/usr/bin/env python
"""Render demos from a ThermTacVLA HDF5 file to MP4.

    python scripts/render_video.py datasets/hdf5/pick_cold_can/pick_cold_can_aluminum_aluminum_lemon_paperboard_aluminum_balsa_iron.hdf5 --demo 0
    # several demos / bigger panels: ... --demo 0 --demo 3 --scale 3

Frames are read straight out of the file, so the video shows exactly what a
dataset consumer sees (not a re-render, which would drift from the recorded
pixels). Both cameras are laid out side by side and upscaled nearest-neighbour;
a header band carries step / phase / gripper / temperature, and contact onsets
from ``events/contact_events`` are marked in the footer strip.

Below that, a **sensing band** draws the fields the tactile and thermal channels
actually contain, synchronised to the same step:
  * one 4x3 |F| map per finger (``obs/tactile/force``, world-frame magnitude —
    rotation-invariant, so no per-link orientation is needed), values printed in
    the cells because a colour alone cannot be read off a still;
  * a fingertip temperature map per finger (``obs/temperature/fingertip_grid``):
    a 4x4 strip map under the engine's default AABB box (docs §9-R26), or the
    lumped ``(1,1,1)`` measured cell of format 1.5 (docs §9-R36) — then there is
    no map to draw, so the panel is skipped and the header readout carries
    **both** fingers' measured temperature;
  * the 4x4x4 object voxel grid (``obs/temperature/object_grid``) as four z
    slices, on a per-episode tight scale — for a good conductor the object is
    very nearly isothermal, and the printed range is what says so;
  * when the file carries ``obs/tactile/proximity`` (docs §9-R23): the same 4x3
    per-taxel map, plus a **time-series strip** — per-finger Σ|F| for both the
    proximity and force channels, which is where the contact-gated force step
    and the ungated proximity ramp can be compared directly. Two vertical
    markers split the interesting interval: cyan at the first nonzero proximity
    (probe sphere touching) and red at the first nonzero depth/force (the rigid
    contact gate opening);
  * a **gel-style imprint** per finger: the contact-height field is upsampled,
    turned into surface normals and lit by three coloured LEDs — the classic
    GelSight look, where hue encodes tilt. The height source, in order, is
    ``obs/tactile/height_map`` (the dense map of format 1.3, which is the marker
    field resampled — and which already spans the marker grid's cells, so the
    panel keeps the true pad aspect), else ``obs/tactile/elastomer`` (the marker
    field itself), else the ``obs/tactile/depth`` probes; the caption names the
    one that was drawn, its size or grid, and the frame it is in. The marker
    field is projected onto the pad normal in the **link frame**, using
    ``obs/link_pose`` to undo the world-frame recording when there is one (a file
    without it falls back to ``|disp|``, which is frame-independent but is not a
    height field). A format-1.6 file records only the in-plane part of the marker
    motion — ``(T, F*P, 2)`` in the pad tangent frame — so there the panel draws
    the in-plane ``|disp|`` and no projection is involved. This *shows* the
    recorded field — it adds no spatial detail
    the field does not have, so a 4x3 source reads as a soft dome while a 9x7 or
    16x16 marker grid reads as a real imprint. With ``--markers`` the lattice is
    overlaid: a dot per undeformed marker and a segment to where it moved — also
    when the dense map is the field drawn, since it was resampled from exactly
    that lattice.

Channel shapes come from the file, not from constants: ``layouts/modality_grids``
says which grid each modality used (4x3 for force/depth, 9x7 or 16x16 for the
marker field), and files written before that attribute render exactly as they
always did through the single ``layouts/probe_grid`` fallback.

Pass ``--no-sensing`` for the camera-only cut. Playback defaults to the policy
rate (20 Hz), i.e. real time. Requires only imageio + Pillow (both in the
tempvla env; imageio_ffmpeg ships its own ffmpeg).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.tools import tactile_field  # noqa: E402  (needs REPO on sys.path)
from sim.tools.gel_render import (   # noqa: E402
    gel_field,
    gel_slope,
    gelsight_image,
    grid_m_per_px,
    lattice_nodes,
    pad_tangents,
)

# Small, readable palette (RGB).
BG = (18, 18, 20)
FG = (235, 235, 235)
DIM = (150, 150, 155)
HOT = (255, 140, 60)      # temperature readouts
COLD = (120, 180, 255)
CONTACT = (255, 70, 70)
OK = (110, 220, 130)

CAMERA_ORDER = ["agentview_rgb", "eye_in_hand_rgb"]
CAMERA_LABEL = {"agentview_rgb": "agentview (fixed)", "eye_in_hand_rgb": "eye-in-hand (wrist)"}

# Colour ramps, as (position, RGB) anchors interpolated per channel. Hand-rolled
# rather than pulled from matplotlib so this tool stays imageio + Pillow only.
RAMP_FORCE = [  # dark -> bright, roughly perceptual: "how hard is this taxel pressed"
    (0.00, (14, 14, 22)), (0.25, (74, 20, 104)), (0.50, (176, 52, 88)),
    (0.75, (242, 132, 24)), (1.00, (252, 250, 168)),
]
RAMP_TEMP = [   # ember: cold iron -> glowing
    (0.00, (28, 32, 46)), (0.30, (110, 44, 112)), (0.62, (216, 84, 34)),
    (0.85, (248, 168, 44)), (1.00, (255, 240, 180)),
]
RAMP_PROX = [   # cool field: nothing -> sensing volume filled (contrast with RAMP_FORCE)
    (0.00, (10, 16, 26)), (0.35, (18, 74, 108)), (0.70, (58, 168, 196)),
    (1.00, (168, 244, 255)),
]
CURVE_L = (110, 220, 130)   # left finger, in both trace rows
CURVE_R = (255, 150, 70)    # right finger
GATE = CONTACT              # rigid-contact gate marker (red, same as contact events)
ONSET = (90, 210, 245)      # first nonzero proximity marker (cyan)


def _ramp(v: np.ndarray, stops) -> np.ndarray:
    """Map ``v`` in [0,1] to uint8 RGB through the anchor ``stops``."""
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
    pos = np.array([s[0] for s in stops], dtype=np.float32)
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    out = np.empty(v.shape + (3,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(v, pos, cols[:, c]).astype(np.uint8)
    return out


def _field_image(values: np.ndarray, cell: int, stops, vmin: float, vmax: float):
    """Heatmap of a (rows, cols) array as a PIL image, nearest-upscaled to ``cell`` px."""
    from PIL import Image

    span = max(float(vmax) - float(vmin), 1e-9)
    rgb = _ramp((np.asarray(values, dtype=np.float32) - vmin) / span, stops)
    rows, cols = values.shape
    return Image.fromarray(rgb).resize((cols * cell, rows * cell), Image.NEAREST), rgb


# The photometric model behind the gel panel (LED rig, height -> normals -> RGB)
# and the grid geometry it needs live in sim/tools/gel_render.py, so any other
# viewer can draw the same picture; what stays here is panel *layout* arithmetic.


def _panel_cell(rows: int, cell: int) -> int:
    """Cell size for a numeric field panel, shrunk so a dense grid still fits the
    band's fixed height. The 4x3 channels keep ``cell`` exactly."""
    return cell if rows <= 4 else max(6, (cell * 4) // rows)


def _block_w(arr, cell: int, gap: int = 12) -> int:
    """Width of a per-finger field block; ``arr`` is (T, fingers, rows, cols)."""
    if arr is None:
        return 0
    return arr.shape[1] * arr.shape[3] * cell + (arr.shape[1] - 1) * gap


def _gel_mpp(demo: dict, k: int, size: tuple[int, int], grid: tuple[int, int]):
    """Metres per output pixel for finger ``k``'s gel panel, or None when the file
    does not hold the probe positions of the grid the drawn field came from.

    The positions are the *gel channel's* (``layouts/grids/<tag>/probe_local_pos``,
    i.e. 16x16 for the marker field), which is not what the ``layouts/`` root alias
    holds once grids differ per modality.
    """
    pp = demo.get("gel_pos")
    if not pp or k >= len(pp) or pp[k].shape[0] != int(grid[0]) * int(grid[1]):
        return None
    return grid_m_per_px(pp[k], (int(grid[0]), int(grid[1])), size)


def _draw_marker_overlay(d, x0: int, y0: int, size: tuple[int, int], tan, disp: np.ndarray,
                         mpp, grid: tuple[int, int], gain: float = 1.0) -> None:
    """Overlay the marker lattice on a gel panel: a dot per undeformed marker, and a
    segment to where that marker actually moved (scaled by ``gain``, since the real
    displacement is a fraction of a pixel at panel size).

    The shading shows only the out-of-plane component; this is the part that shows
    the recorded *vector* field — what a marker tracker measures, and what makes the
    in-plane shear visible at all.

    ``disp`` is either (rows, cols, 3) link-frame displacement, projected here onto
    the pad tangent axes ``tan``, or (rows, cols, 2) pad-tangent displacement
    (format 1.6) — already in the image frame, component 1 across the columns and
    component 0 down the rows, so it needs no tangent basis.
    """
    if mpp is None or disp is None or (disp.shape[-1] == 3 and tan is None):
        return
    w, h = int(size[0]), int(size[1])
    xs, ys = lattice_nodes((w, h), grid)
    if disp.shape[-1] == 2:
        px = xs + gain * disp[..., 1] / mpp[0]      # image x = grid columns
        py = ys + gain * disp[..., 0] / mpp[1]      # image y = grid rows
    else:
        u_col, u_row = tan
        px = xs + gain * (disp @ u_col) / mpp[0]      # image x = grid columns
        py = ys + gain * (disp @ u_row) / mpp[1]      # image y = grid rows
    for r in range(xs.shape[0]):
        for c in range(xs.shape[1]):
            sx, sy = x0 + int(xs[r, c]), y0 + int(ys[r, c])
            nx, ny = x0 + int(px[r, c]), y0 + int(py[r, c])
            if abs(nx - sx) + abs(ny - sy) < 1:
                d.point((sx, sy), fill=(205, 205, 215))
            else:
                d.line([sx, sy, nx, ny], fill=(255, 240, 120), width=1)
                d.point((nx, ny), fill=(255, 240, 120))


def _cell_numbers(d, x: int, y: int, values: np.ndarray, rgb: np.ndarray, cell: int, font, fmt: str) -> None:
    """Print each value centred in its own cell, in whichever of black/white reads."""
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            txt = fmt.format(float(values[r, c]))
            w = d.textlength(txt, font=font)
            bg = rgb[r, c].astype(np.float32)
            luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            d.text(
                (x + c * cell + (cell - w) / 2, y + r * cell + (cell - 15) / 2),
                txt, font=font, fill=(10, 10, 12) if luma > 150 else (240, 240, 245),
            )


def _trace_row(d, x: int, y: int, w: int, h: int, ts, t: int, n: int, vmax: float,
               label: str, onset_step, gate_step, font) -> None:
    """One mini time-series row: per-finger curves over all steps + a time cursor.

    ``ts`` is (T, 2) — 0 = left finger, 1 = right. Markers: cyan = first nonzero
    proximity, red = the rigid contact gate opening (first nonzero depth/force).
    """
    d.rectangle([x, y, x + w, y + h], fill=(24, 24, 28), outline=(64, 64, 70))
    d.line([x + 1, y + h - 1, x + w - 1, y + h - 1], fill=(70, 70, 76))  # zero baseline
    if ts is not None:
        for k, col in ((0, CURVE_L), (1, CURVE_R)):
            if ts.shape[1] <= k:
                continue
            pts = [
                (x + 1 + (w - 2) * (i / max(n - 1, 1)),
                 y + h - 2 - (h - 4) * min(float(ts[i, k]) / vmax, 1.0))
                for i in range(n)
            ]
            if len(pts) > 1:
                d.line(pts, fill=col, width=1)
    for marker, col in ((onset_step, ONSET), (gate_step, GATE)):
        if marker is not None:
            mx = x + 1 + (w - 2) * (marker / max(n - 1, 1))
            d.line([mx, y, mx, y + h], fill=col)
    vals = "" if ts is None else "   " + "  ".join(f"{'LR'[k]} {ts[t, k]:.2f}" for k in range(min(2, ts.shape[1])))
    txt = f"{label}{vals}"
    d.rectangle([x + 1, y + 1, x + 5 + d.textlength(txt, font=font), y + 15], fill=(24, 24, 28))
    d.text((x + 4, y + 2), txt, font=font, fill=(225, 225, 230))
    cx = x + 1 + (w - 2) * (t / max(n - 1, 1))
    d.line([cx, y, cx, y + h], fill=FG)  # current step


class UnknownDemo(Exception):
    """A --demo index that this file does not contain."""


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10
    except TypeError:
        return ImageFont.load_default()


def _fmt_temp(v: float | None) -> str:
    return "--" if v is None else f"{v:5.1f} C"


def load_demo(path: Path, demo: str) -> dict:
    """Pull everything the video needs out of one demo, into plain arrays."""
    with h5py.File(path, "r") as f:
        if demo not in f["data"]:
            have = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            raise UnknownDemo(
                f"{path} has no {demo} — it holds {len(have)} demos "
                f"({have[0]}..{have[-1]}); pass --demo with one of those indices"
            )
        g = f["data"][demo]
        obs = g["obs"]
        # Layouts first: the field panels need each channel's probe grid, the probe
        # frame (metres per pixel, from the probe spacing) and the pad normals.
        # New files say which grid each modality used (`layouts/modality_grids`,
        # e.g. 4x3 for force and 16x16 for the marker field); older files carry one
        # `layouts/probe_grid` for everything, which is what tactile_field falls
        # back to — so the same code reads both.
        grids: dict[str, tuple[int, int]] = {}
        geom: dict[str, tuple[list, list | None]] = {}
        for m in ("force", "proximity", "depth", "elastomer"):
            grid_m = tactile_field.modality_grid(f, m)
            if grid_m is not None:
                grids[m] = (int(grid_m[0]), int(grid_m[1]))
            pos_m, nrm_m = tactile_field.finger_geometry(f, m)
            if pos_m:
                geom[m] = (pos_m, nrm_m)
        out = {
            "attrs": dict(g.attrs),
            "cameras": {},
            "gripper": obs["gripper_states"][:, 0].astype(np.float32) if "gripper_states" in obs else None,
            "object_temp": obs["temperature/object"][:, 0].astype(np.float32) if "temperature/object" in obs else None,
            "fingertip": None,
            "fingertip_lumped": None,  # (T, F) measured cell when the grid is (1,1,1)
            "ee_pos": obs["ee_pos"][:].astype(np.float32) if "ee_pos" in obs else None,
            "contact_steps": [],
            # fields for the sensing band
            "force_grid": None,     # (T, F, R, C) |F| per taxel, split by finger
            "prox_grid": None,      # (T, F, R, C) |F| per taxel (proximity)
            "force_ts": None,       # (T, F) Σ|F| over a finger's R*C taxels
            "prox_ts": None,        # (T, F) Σ|F| (proximity is on its own scale)
            "depth_ts": None,       # (T, F) max depth per finger, m
            "depth_grid": None,     # (T, F, R, C) probe penetration depth, m
            "fingertip_grid": None,  # (T, F, rows, cols) C — the finger pad map
            "object_grid": None,    # (T, 4, 4, 4) C
            "finger_names": ["finger 0", "finger 1"],
            "grids": grids,         # modality -> (rows, cols), as recorded
            "geom": geom,           # modality -> (per-finger probe_local_pos, normals)
            # gel-style panel: a per-finger contact-height field, in metres, 0 = out of contact
            "gel_grid": None,       # (T, F, R, C)
            "gel_src": None,        # what that field is — printed in the caption
            "gel_hmax": None,       # per-episode ceiling, so the imprint is comparable frame to frame
            "gel_cells": None,      # (rows, cols) of pad *cells* the drawn panel spans
            "gel_pos": None,        # per finger (P, 3) positions of the grid that field came from
            "gel_disp": None,       # (T, F, R, C, 2|3) marker displacement, m; 2 = pad tangent
            "gel_tan": None,        # per finger (u_col, u_row) — image axes of a 3-component field
        }
        if "temperature/fingertip_grid" in obs:
            ft = obs["temperature/fingertip_grid"][:].astype(np.float32)
            out["fingertip"] = ft.reshape(ft.shape[0], -1).max(axis=1)  # hottest taxel per step
            # Format 1.5: the lumped (1,1,1) measured cell — one value per finger,
            # no spatial axis to draw. Keep the per-finger pair for the header.
            if int(np.prod(ft.shape[2:])) == 1:
                out["fingertip_lumped"] = ft.reshape(ft.shape[0], ft.shape[1])
            # (T, fingers, nx, ny, nz) -> (T, fingers, rows, cols): an AABB finger
            # grid (docs §9-R26) has exactly two live spatial axes and one collapsed
            # normal slab (nz=1), and keeps the recorded axis order.
            live = [s for s in ft.shape[2:] if s > 1]
            if ft.ndim == 5 and len(live) == 2:
                out["fingertip_grid"] = ft.reshape(ft.shape[0], ft.shape[1], *live)
        if "temperature/object_grid" in obs:
            og = obs["temperature/object_grid"][:].astype(np.float32)
            out["object_grid"] = og if og.ndim == 4 else None
        # Tactile channels: rows are (finger, then that modality's grid), so the
        # reshape is driven by the grid recorded in layouts/ — a channel whose row
        # count is not a whole number of grids is skipped rather than mis-shapen.
        if "tactile/force" in obs:
            F = obs["tactile/force"][:].astype(np.float32)  # (T, F*R*C, 3) world frame
            grid = grids.get("force")
            n_p = grid[0] * grid[1] if grid else 0
            if n_p and F.shape[1] % n_p == 0:
                nf = F.shape[1] // n_p
                mag = np.linalg.norm(F, axis=2)             # rotation-invariant
                out["force_grid"] = mag.reshape(F.shape[0], nf, *grid)  # index = len*C + wid
                out["force_ts"] = mag.reshape(F.shape[0], nf, n_p).sum(axis=2)
        if "tactile/proximity" in obs:
            P = obs["tactile/proximity"][:].astype(np.float32)
            grid = grids.get("proximity")
            n_p = grid[0] * grid[1] if grid else 0
            if n_p and P.shape[1] % n_p == 0:
                nf = P.shape[1] // n_p
                pmag = np.linalg.norm(P, axis=2)
                out["prox_grid"] = pmag.reshape(P.shape[0], nf, *grid)
                out["prox_ts"] = pmag.reshape(P.shape[0], nf, n_p).sum(axis=2)
        if "tactile/depth" in obs:
            D = obs["tactile/depth"][:].astype(np.float32)
            grid = grids.get("depth")
            n_p = grid[0] * grid[1] if grid else 0
            if n_p and D.shape[1] % n_p == 0:
                nf = D.shape[1] // n_p
                out["depth_grid"] = D.reshape(D.shape[0], nf, *grid)   # (along, across)
                out["depth_ts"] = D.reshape(D.shape[0], nf, n_p).max(axis=2)
        # Height source for the gel-style panel, in priority order: the recorded
        # dense height map (format 1.3 — the marker field resampled onto a raster),
        # else the marker field projected onto that grid's pad normal, else the
        # coarse depth field. Either way the caption says which one was drawn.
        #
        # The marker extras (`gel_disp` / `gel_pos` / `gel_tan`, below) are filled
        # whenever the marker channel is present, even when the height map is the
        # field actually drawn: the --markers overlay tracks the *marker lattice*,
        # and the map spans exactly that lattice's cells, so the overlay lands on
        # the markers it was resampled from.
        height_map_cells = None
        if "tactile/height_map" in obs:
            HM = obs["tactile/height_map"][:].astype(np.float32)   # (T, F, H, W) m
            if HM.ndim == 4 and HM.shape[0]:
                field = HM
                # Sign convention, same as the coarse field below: the pad normal
                # points out of the gel, so an indentation reads negative. Draw the
                # deformation as positive height — a bulge and an equal dent differ
                # only by hue order, and "pressed into the gel" is the imprint to show.
                if float(np.abs(field.min())) > float(np.abs(field.max())):
                    field = -field
                if float(np.abs(field).max()) > 0.0:
                    out["gel_grid"] = field
                    out["gel_src"] = f"height_map {field.shape[2]}x{field.shape[3]} (marker field resampled)"
                    grid = grids.get("elastomer")
                    height_map_cells = tuple(grid) if grid else None
        # Marker field: the vector channel behind a height map, and the height
        # source itself when no map was recorded.
        if "tactile/elastomer" in obs:
            # (T, F*P, C) marker displacement, m — fingers concatenated along the probe
            # axis (same layout as the force channel), P = that grid's rows*cols.
            # C=2 (format 1.6) is pad-tangent (u = rows, v = cols); C=3 is the older
            # link/world-frame vector that still carries the out-of-plane part.
            E = obs["tactile/elastomer"][:].astype(np.float32)
            grid = grids.get("elastomer")
            n_p = grid[0] * grid[1] if grid else 0
            if n_p and E.shape[1] % n_p == 0 and E.shape[-1] in (2, 3):
                nf = E.shape[1] // n_p
                n_comp = int(E.shape[-1])
                Ef = E.reshape(E.shape[0], nf, n_p, n_comp)
                Eg = Ef.reshape(E.shape[0], nf, *grid, n_comp)   # per-marker, for the projection
                pos, nrm = geom.get("elastomer", (None, None))
                if n_comp == 2:
                    # Format 1.6: displacement in the pad's tangent plane, (u = grid
                    # rows, v = grid columns), out-of-plane dropped. There is no
                    # height to project onto the normal, so the panel draws the
                    # in-plane magnitude, and the marker overlay works on (u, v)
                    # directly (see _draw_marker_overlay).
                    field, src = np.linalg.norm(Ef, axis=-1), "marker |disp| pad tangent"
                    out["gel_disp"] = Eg.astype(np.float64)
                else:
                    # A gel frame is the *out-of-plane* deformation, and "out of plane" is
                    # the pad normal — not an axis of the link frame (on the Panda fingers
                    # it is -y), so the projection needs the stored normal *and* the link
                    # frame. World-frame recordings are rotated back with obs/link_pose,
                    # which makes that exact; a file predating link_pose has neither the
                    # pose nor (usually) an addressable normal, and there |disp| is the
                    # only frame-independent quantity left.
                    nrm_f = (np.stack([n.reshape(*grid, 3) for n in nrm])
                             if nrm is not None and len(nrm) == nf else None)
                    if nrm_f is None:
                        field, src = np.linalg.norm(Ef, axis=-1), "gel |disp| (no pad normal)"
                    elif not tactile_field.world_frame(f):
                        field = (Eg * nrm_f[None]).sum(axis=-1)
                        src = "marker disp·n  link frame (recorded)"
                        out["gel_disp"] = Eg.astype(np.float64)
                    elif "link_pose" not in obs:
                        field, src = np.linalg.norm(Ef, axis=-1), "gel |disp|  world frame, no link_pose"
                    else:
                        disp = tactile_field.read_markers(f, demo, "elastomer", frame="link")
                        field = (disp * nrm_f[None]).sum(axis=-1)
                        src = "marker disp·n  link frame via link_pose"
                        out["gel_disp"] = disp
                # Sign convention: the normal points out of the pad, so an
                # indentation reads negative. Draw the deformation as positive
                # height — a bulge and an equal dent differ only by hue order, and
                # "pressed into the gel" is the imprint to show.
                if float(np.abs(field.min())) > float(np.abs(field.max())):
                    field = -field
                if float(np.abs(field).max()) > 0.0:
                    # The field/its caption only when nothing of higher priority (the
                    # recorded height map) is already drawn; the positions and the
                    # marker vectors are always kept — the overlay needs them.
                    if out["gel_grid"] is None:
                        out["gel_grid"] = field.reshape(E.shape[0], nf, *grid)
                        out["gel_src"] = f"{src} {grid[0]}x{grid[1]}"
                    out["gel_pos"] = pos
                    if (n_comp == 3 and out["gel_disp"] is not None and pos is not None):
                        out["gel_tan"] = [pad_tangents(p, tuple(grid)) for p in pos]
        if out["gel_grid"] is None and out["depth_grid"] is not None:
            dg = out["depth_grid"]
            out["gel_grid"] = dg
            out["gel_src"] = f"depth ({dg.shape[2] * dg.shape[3]} probes)"
            out["gel_pos"] = geom.get("depth", (None, None))[0]
            # The marker overlay tracks the marker lattice, and this panel is the
            # coarse probe grid: a marker field kept alongside it (possible only
            # when it was identically zero, so there is nothing to draw) must not
            # be overlaid on the wrong lattice.
            out["gel_disp"] = None
            out["gel_tan"] = None
        if out["gel_grid"] is not None:
            out["gel_hmax"] = max(float(np.abs(out["gel_grid"]).max()), 1e-12)
            # Physical extent of the drawn panel, in pad *cells*: what the m/px and
            # the marker lattice are derived from. The height map spans the marker
            # grid's cells (tools/height_map.py); every other source spans its own
            # array's shape. Same model as gel_render.grid_m_per_px, so the overlay
            # and the shading agree on both.
            out["gel_cells"] = height_map_cells or tuple(out["gel_grid"].shape[2:])
        if "layouts/link_names" in f:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["layouts/link_names"][:]]
            short = []
            for n in names:
                low = n.lower()
                short.append("L" if "left" in low else "R" if "right" in low else n.replace("panda_", "")[:5])
            out["finger_names"] = short or out["finger_names"]
        for cam in CAMERA_ORDER:
            if cam in obs:
                out["cameras"][cam] = obs[cam][:]  # (T,H,W,3) uint8 — held in RAM (few MB)
        if "events/contact_events/start_step" in g:
            out["contact_steps"] = [int(s) for s in g["events/contact_events/start_step"][:]]
    if not out["cameras"]:
        raise SystemExit(f"{path} has no RGB channels — regenerate with --cameras (docs §8).")
    lengths = {c: v.shape[0] for c, v in out["cameras"].items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"camera lengths disagree: {lengths}")
    out["steps"] = next(iter(lengths.values()))
    return out


def compose(demo: dict, path: Path, demo_name: str, scale: int, fps: float, stride: int = 1,
            sensing: bool = True, markers: bool = False):
    """Yield composited RGB frames (H,W,3) uint8, one per kept policy step."""
    from PIL import Image, ImageDraw

    cams = demo["cameras"]
    names = [c for c in CAMERA_ORDER if c in cams]
    h, w = cams[names[0]].shape[1:3]
    scale = max(1, int(scale))
    iw, ih = w * scale, h * scale

    pad = 6
    header = 74
    footer = 26

    # Sensing band: force maps are the tallest panel, so they set the band height.
    F_CELL, T_CELL, O_CELL = 28, 22, 16
    band = bool(sensing and (demo["force_grid"] is not None or demo["fingertip_grid"] is not None
                             or demo["prox_grid"] is not None or demo["gel_grid"] is not None))

    # Per-episode colour scales. Force keeps a fixed 0..5 N ceiling so episodes
    # are comparable; temperature adapts, and both bounds get printed — the
    # object grid is nearly isothermal for a metal, and only the numbers say so.
    fg = demo["force_grid"]
    f_max = 5.0
    ftg = demo["fingertip_grid"]
    ft_lo = float(ftg.min()) if ftg is not None else 0.0
    ft_hi = float(ftg.max()) if ftg is not None else 1.0
    og = demo["object_grid"]
    og_lo = float(og.min()) if og is not None else 0.0
    og_hi = float(og.max()) if og is not None else 1.0
    if og is not None and (og_hi - og_lo) < 1e-6:
        og_hi = og_lo + 1e-6

    # Proximity has its own (much smaller, differently calibrated) scale — the
    # point-cloud sum is not N-for-N comparable with KinematicTaxel (docs §9-R23).
    pg = demo["prox_grid"]
    prox_vmax = max(float(demo["prox_ts"].max()) * 1.05, 1e-6) if demo["prox_ts"] is not None else 1.0
    # The per-taxel maps need their own ceiling: scaling a single cell against the
    # 12-taxel sum would leave every cell dark.
    prox_cell_vmax = max(float(pg.max()) * 1.05, 1e-6) if pg is not None else 1.0
    force_vmax = max(float(demo["force_ts"].max()) * 1.05, 1.0) if demo["force_ts"] is not None else 1.0
    trace = bool(band and (demo["force_ts"] is not None or demo["prox_ts"] is not None))

    # The two numeric field blocks (force, proximity) take their shape from the
    # file's own grid, and their cells shrink if a channel ever gets denser than
    # 4x3 — a per-taxel readout with a number in every cell only works coarse.
    FC_F = _panel_cell(fg.shape[2], F_CELL) if fg is not None else F_CELL
    FC_P = _panel_cell(pg.shape[2], F_CELL) if pg is not None else F_CELL
    # Fingertip temperature panel: a finger grid denser than 4x4 shrinks its
    # cells to the band's height (F_CELL is the tallest panel's base) instead of
    # the 22 px a 4x4 AABB grid uses. A cell below FTG_NUM_PX cannot hold its
    # printed number, and the map is read from the colour ramp alone.
    FTG_NUM_PX = 14
    if ftg is not None:
        FTG_ROWS, FTG_COLS = int(ftg.shape[2]), int(ftg.shape[3])
        FTG_CELL = T_CELL if FTG_ROWS <= 4 else _panel_cell(FTG_ROWS, F_CELL)
    else:
        FTG_ROWS = FTG_COLS = 0
        FTG_CELL = T_CELL
    FTG_GAP = 8
    FTG_W = ftg.shape[1] * FTG_COLS * FTG_CELL + (ftg.shape[1] - 1) * FTG_GAP if ftg is not None else 0
    FTG_H = FTG_ROWS * FTG_CELL

    def _first_nonzero(ts):
        if ts is None:
            return None
        nz = np.nonzero(ts.max(axis=1) > 0)[0]
        return int(nz[0]) if nz.size else None

    onset_step = _first_nonzero(demo["prox_ts"])
    gate_step = _first_nonzero(demo["depth_ts"] if demo["depth_ts"] is not None else demo["force_ts"])
    trace_title = "Σ|F| per finger — proximity (cyan onset) vs force (red gate)"
    TRACE_W = 420

    # Gel-style panel: a photometric render of the contact-height field, kept the
    # band's full height so it is the same size as a force map column-for-column
    # (the pad is deeper along the finger than across it, hence the aspect).
    gg = demo["gel_grid"]
    GEL_H = 4 * F_CELL
    # Width from the panel's *physical* cell extent, not the array's shape: a stored
    # height map is a square 64x64 raster over a 9x7-cell pad, and drawing that at
    # array aspect would stretch the imprint against every other panel.
    gel_cells = tuple(demo["gel_cells"] or gg.shape[2:]) if gg is not None else (1, 1)
    GEL_W = max(48, round(GEL_H * (gel_cells[1] / gel_cells[0]))) if gg is not None else 0
    GEL_GAP = 8
    # Slopes in a recorded gel field are tiny (a HydroShear pad moves tens of µm
    # over a centimetre: ~1e-3 m/m, invisible at true scale), so the shading is
    # exaggerated by one factor per episode — the steepest slope of the peak frame
    # is brought to a readable tilt and the factor is printed in the caption.
    GEL_TARGET_SLOPE = 0.6
    gel_gain = 1.0
    if gg is not None:
        t_peak = int(np.argmax(gg.reshape(gg.shape[0], -1).max(axis=1)))
        need = []
        for k in range(gg.shape[1]):
            mpp = _gel_mpp(demo, k, (GEL_W, GEL_H), gel_cells)
            if mpp is None:
                continue
            hr, sc = gel_field(gg[t_peak, k], (GEL_W, GEL_H), hmax=demo["gel_hmax"])
            gx, gy = gel_slope(hr, sc, mpp)
            need.append(float(max(np.abs(gx).max(), np.abs(gy).max())))
        if need and max(need) > 0:
            gel_gain = min(1e6, max(1.0, GEL_TARGET_SLOPE / max(need)))

    # Marker overlay (--markers): a HydroShear pad displaces tens of µm over a
    # centimetre of gel, which at this panel scale is well under a pixel, so the
    # vectors get scaled by one factor per episode and the caption prints it —
    # the same treatment the shading gets.
    MARKER_TARGET_PX = 3.5
    marker_gain = 1.0
    if markers and gg is not None and demo["gel_disp"] is not None:
        need_px = []
        for k in range(min(gg.shape[1], demo["gel_disp"].shape[1])):
            mpp = _gel_mpp(demo, k, (GEL_W, GEL_H), gg.shape[2:])
            if mpp is None:
                continue
            dk = demo["gel_disp"][:, k]
            if dk.shape[-1] == 2:
                # pad tangent: component 1 crosses image x, component 0 image y
                need_px.append(float(max(np.abs(dk[..., 1]).max() / mpp[0],
                                         np.abs(dk[..., 0]).max() / mpp[1])))
            elif demo["gel_tan"] is not None:
                u_col, u_row = demo["gel_tan"][k]
                need_px.append(float(max(np.abs(dk @ u_col).max() / mpp[0],
                                         np.abs(dk @ u_row).max() / mpp[1])))
        if need_px and max(need_px) > 0:
            marker_gain = min(1000.0, max(1.0, MARKER_TARGET_PX / max(need_px)))

    f_title = _font(17)
    f_body = _font(15)
    f_small = _font(13)
    f_cam = _font(13)
    f_cell_f = _font(10)   # "4.5" must fit a 28 px cell without touching its neighbour
    f_cell_t = _font(11)
    f_cap = _font(11)

    attrs = demo["attrs"]
    material = str(attrs.get("object_material", "?"))
    seed = attrs.get("episode_seed", "?")
    success = bool(attrs.get("success", False))
    final_phase = str(attrs.get("expert_final_phase", "?"))
    n = demo["steps"]

    # Section x positions, and the object title text (needed to size the canvas).
    # Left to right: the two tactile maps, then their shared time series, then thermal.
    # A scratch drawer measures text before anything is placed — some titles are
    # wider than their panels, and the next section has to clear the wider of the two.
    _scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x_force = pad
    x_prox = x_force + (_block_w(fg, FC_F) if fg is not None else 0)
    x_trace = x_prox + (_block_w(pg, FC_P) if pg is not None else 0) + (16 if trace else 0)
    x_temp = x_trace + (TRACE_W + 16 if trace else 0)
    x_obj = x_temp + (FTG_W if ftg is not None else 0) + (8 if ftg is not None and og is not None else 0)
    obj_title = (f"object (C)  {og_lo:.1f}..{og_hi:.1f}  span {og_hi - og_lo:.2f} (stretched)"
                 if og is not None else "")
    x_gel = int(x_obj + ((max(4 * (4 * O_CELL + 5), _scratch.textlength(obj_title, font=f_cap)) + 12)
                         if og is not None else 0))
    gel_title = (f"gel-style imprint — {demo['gel_src']}"
                 + (f"  slope x{gel_gain:.0f}" if gel_gain > 1.5 else "")
                 + (f"  vectors x{marker_gain:.0f}"
                    if markers and marker_gain > 1.5 and demo["gel_disp"] is not None else "")
                 ) if gg is not None else ""
    head_text = f"{path.name}  ·  {demo_name}  ·  {material}  ·  seed {seed}"

    # The band and the header have fixed pixel sizes that do not shrink with
    # --scale, so the canvas is widened to fit whichever needs the most room
    # (at --scale 2 both used to run off the right edge). Titles count too —
    # the object caption is wider than the four z-slices under it.
    band_right = 0.0
    if fg is not None:
        band_right = max(band_right, x_force + fg.shape[1] * fg.shape[3] * FC_F,
                         x_force + _scratch.textlength("tactile |F| (N)", font=f_cap))
    if ftg is not None:
        band_right = max(band_right, x_temp + FTG_W,
                         x_temp + _scratch.textlength(f"fingertip temp (C)  {ft_lo:.0f}..{ft_hi:.0f}", font=f_cap))
    if og is not None:
        band_right = max(band_right, x_obj + 4 * (4 * O_CELL + 5),
                         x_obj + _scratch.textlength(obj_title, font=f_cap))
    if pg is not None:
        band_right = max(band_right, x_prox + pg.shape[1] * pg.shape[3] * FC_P,
                         x_prox + _scratch.textlength("tactile proximity |F| (a.u.)", font=f_cap))
    if trace:
        band_right = max(band_right, x_trace + TRACE_W,
                         x_trace + _scratch.textlength(trace_title, font=f_cap))
    if gg is not None:
        band_right = max(band_right, x_gel + 2 * GEL_W + GEL_GAP,
                         x_gel + _scratch.textlength(gel_title, font=f_cap))
    W = int(max(
        len(names) * iw + (len(names) + 1) * pad,
        band_right + pad if band else 0,
        pad + _scratch.textlength(head_text, font=f_title) + 26,
    ))
    if W % 2:                      # libx264 + yuv420p wants even dimensions
        W += 1
    H = header + ih + footer + pad
    band_top = H
    H += (20 + 4 * F_CELL + 20) if band else 0

    for t in range(0, n, stride):
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)

        # --- header ------------------------------------------------------
        d.text((pad, 8), head_text, font=f_title, fill=FG)
        # a dot for success, so a failed take is obvious at a glance
        d.ellipse([W - pad - 14, 12, W - pad - 4, 22], fill=OK if success else CONTACT)

        grip = None if demo["gripper"] is None else float(demo["gripper"][t])
        if grip is None:
            grip_s = "grip   --"
        else:
            state = "open" if grip < -0.5 else ("closed" if grip > -0.1 else "closing")
            grip_s = f"grip {grip:+.2f} ({state})"
        t_obj = None if demo["object_temp"] is None else float(demo["object_temp"][t])
        t_fin = None if demo["fingertip"] is None else float(demo["fingertip"][t])
        z = None if demo["ee_pos"] is None else float(demo["ee_pos"][t, 2])

        d.text((pad, 32), f"step {t:>3}/{n}   t={t / fps:5.2f}s   {grip_s}", font=f_body, fill=FG)

        # Readouts flow left to right with measured widths, so a label can never
        # overlap its value (monospace advance varies with the font that's found).
        def field(x: float, label: str, value: str, color) -> float:
            d.text((x, 52), label, font=f_small, fill=DIM)
            x += d.textlength(label, font=f_small) + 5
            d.text((x, 51), value, font=f_body, fill=color)
            return x + d.textlength(value, font=f_body) + 18

        x = pad
        x = field(x, "object", _fmt_temp(t_obj), HOT)
        fl = demo["fingertip_lumped"]
        if fl is not None:
            # Format 1.5: one measured value per finger — print the pair, not
            # the cross-finger max (the two fingers run 5-15 C apart, docs R33).
            lv, rv = float(fl[t, 0]), float(fl[t, 1])
            x = field(x, "fingertip L/R", f"{lv:5.1f} / {rv:5.1f} °C",
                      HOT if max(lv, rv) > 30 else COLD)
        else:
            x = field(x, "fingertip", _fmt_temp(t_fin), HOT if (t_fin or 0) > 30 else COLD)
        x = field(x, "tcp z", "--" if z is None else f"{z:.3f} m", FG)
        d.text((x, 52), f"final phase: {final_phase}", font=f_small, fill=DIM)

        # --- cameras -----------------------------------------------------
        contact_now = t in demo["contact_steps"]
        for i, name in enumerate(names):
            x0 = pad + i * (iw + pad)
            frame = cams[name][t]
            img = Image.fromarray(frame).resize((iw, ih), Image.NEAREST)
            canvas.paste(img, (x0, header))
            label = CAMERA_LABEL.get(name, name)
            d.rectangle([x0, header - 1, x0 + iw - 1, header + ih], outline=CONTACT if contact_now else (60, 60, 64))
            d.text((x0 + 4, header + ih - 16), label, font=f_cam, fill=(255, 255, 255))
            d.text((x0 + 4, header + 4), f"{w}x{h}", font=f_cam, fill=(210, 210, 210))

        # --- footer: step ruler with contact onsets ----------------------
        y = header + ih + 8
        d.line([pad, y + 4, W - pad, y + 4], fill=(70, 70, 74))
        for s in demo["contact_steps"]:
            xs = pad + (W - 2 * pad - 1) * (s / max(n - 1, 1))
            d.line([xs, y - 1, xs, y + 9], fill=CONTACT)
            d.text((xs + 3, y - 6), f"contact@{s}", font=f_small, fill=CONTACT)
        xt = pad + (W - 2 * pad - 1) * (t / max(n - 1, 1))
        d.line([xt, y - 3, xt, y + 11], fill=FG)

        # --- sensing band: what the tactile / thermal channels hold -------
        # Positions are computed before anything is drawn: the three sections
        # have different panel heights and widths, and hard-coded offsets had
        # their titles and captions colliding.
        if band:
            y_title = band_top + 3
            y_panel = band_top + 20

            if fg is not None:
                d.text((x_force, y_title), "tactile |F| (N)", font=f_cap, fill=DIM)
                f_rows, f_cols = fg.shape[2], fg.shape[3]
                for k in range(fg.shape[1]):
                    vals = fg[t, k]                                  # (rows, cols) = (along, across)
                    xk = x_force + k * (f_cols * FC_F + 12)
                    img, rgb = _field_image(vals, FC_F, RAMP_FORCE, 0.0, f_max)
                    canvas.paste(img, (xk, y_panel))
                    if FC_F >= 20:            # a smaller cell cannot hold a printed number
                        _cell_numbers(d, xk, y_panel, vals, rgb, FC_F, f_cell_f, "{:.1f}")
                    d.rectangle([xk - 1, y_panel - 1, xk + f_cols * FC_F, y_panel + f_rows * FC_F],
                                outline=(64, 64, 70))
                    nm = demo["finger_names"][k] if k < len(demo["finger_names"]) else f"F{k}"
                    d.text((xk, y_panel + f_rows * FC_F + 3), f"{nm} peak {vals.max():.2f} N",
                           font=f_cap, fill=DIM)

            if ftg is not None:
                d.text((x_temp, y_title), f"fingertip temp (C)  {ft_lo:.0f}..{ft_hi:.0f}",
                       font=f_cap, fill=DIM)
                for k in range(ftg.shape[1]):
                    vals = ftg[t, k]
                    xk = x_temp + k * (FTG_COLS * FTG_CELL + FTG_GAP)
                    img, rgb = _field_image(vals, FTG_CELL, RAMP_TEMP, ft_lo, ft_hi)
                    canvas.paste(img, (xk, y_panel))
                    if FTG_CELL >= FTG_NUM_PX:
                        _cell_numbers(d, xk, y_panel, vals, rgb, FTG_CELL, f_cell_t, "{:.0f}")
                    d.rectangle([xk - 1, y_panel - 1, xk + FTG_COLS * FTG_CELL, y_panel + FTG_H],
                                outline=(64, 64, 70))
                    # Same finger order as the tactile layout: build_finger_temperature_sensors
                    # walks cfg.robot.finger_link_names, which is what layouts/link_names records.
                    nm = demo["finger_names"][k] if k < len(demo["finger_names"]) else f"F{k}"
                    d.text((xk, y_panel + FTG_H + 3), f"tip {nm}", font=f_cap, fill=DIM)

            if og is not None:
                # The tight scale is deliberate — a metal object is very nearly
                # isothermal, so a fixed scale would show one flat colour — but
                # the span has to stay on screen or the contrast reads as heat.
                d.text((x_obj, y_title), obj_title, font=f_cap, fill=DIM)
                for k in range(og.shape[1]):
                    vals = og[t, k]
                    xk = x_obj + k * (4 * O_CELL + 5)
                    img, _ = _field_image(vals, O_CELL, RAMP_TEMP, og_lo, og_hi)
                    canvas.paste(img, (xk, y_panel))
                    d.rectangle([xk - 1, y_panel - 1, xk + 4 * O_CELL, y_panel + 4 * O_CELL],
                                outline=(64, 64, 70))
                    d.text((xk, y_panel + 4 * O_CELL + 3), f"z{k}", font=f_cap, fill=DIM)

            if gg is not None:
                # A photometric (GelSight-style) render of the contact-height field:
                # hue = surface tilt, brightness = tilt toward that LED. No new
                # spatial detail — the caption names the field, its grid and its frame.
                d.text((x_gel, y_title), gel_title, font=f_cap, fill=DIM)
                for k in range(gg.shape[1]):
                    vals = gg[t, k]                                   # (rows, cols) metres
                    xk = x_gel + k * (GEL_W + GEL_GAP)
                    mpp = _gel_mpp(demo, k, (GEL_W, GEL_H), gel_cells)
                    img = gelsight_image(vals, (GEL_W, GEL_H), mpp, hmax=demo["gel_hmax"], gain=gel_gain)
                    canvas.paste(img, (xk, y_panel))
                    if markers:
                        _draw_marker_overlay(
                            d, xk, y_panel, (GEL_W, GEL_H),
                            None if demo["gel_tan"] is None else demo["gel_tan"][k],
                            None if demo["gel_disp"] is None else demo["gel_disp"][t, k],
                            mpp, gel_cells, marker_gain,
                        )
                    d.rectangle([xk - 1, y_panel - 1, xk + GEL_W, y_panel + GEL_H],
                                outline=(64, 64, 70))
                    nm = demo["finger_names"][k] if k < len(demo["finger_names"]) else f"F{k}"
                    # Marker fields are µm-scale (docs §9-R25): "%.2f mm" would
                    # round a 4.9 µm dome to 0.00, so switch units below 0.1 mm.
                    peak = float(vals.max())
                    peak_txt = f"{peak * 1e6:.1f} µm" if peak < 1e-4 else f"{peak * 1e3:.2f} mm"
                    d.text((xk, y_panel + GEL_H + 3), f"{nm} max {peak_txt}",
                           font=f_cap, fill=DIM)

            if pg is not None:
                d.text((x_prox, y_title), "tactile proximity |F| (a.u.)", font=f_cap, fill=DIM)
                p_rows, p_cols = pg.shape[2], pg.shape[3]
                for k in range(pg.shape[1]):
                    vals = pg[t, k]
                    xk = x_prox + k * (p_cols * FC_P + 12)
                    img, rgb = _field_image(vals, FC_P, RAMP_PROX, 0.0, prox_cell_vmax)
                    canvas.paste(img, (xk, y_panel))
                    if FC_P >= 20:
                        _cell_numbers(d, xk, y_panel, vals, rgb, FC_P, f_cell_f, "{:.2f}")
                    d.rectangle([xk - 1, y_panel - 1, xk + p_cols * FC_P, y_panel + p_rows * FC_P],
                                outline=(64, 64, 70))
                    nm = demo["finger_names"][k] if k < len(demo["finger_names"]) else f"F{k}"
                    d.text((xk, y_panel + p_rows * FC_P + 3), f"{nm} Σ {demo['prox_ts'][t, k]:.2f}",
                           font=f_cap, fill=DIM)

            if trace:
                d.text((x_trace, y_title), trace_title, font=f_cap, fill=DIM)
                _trace_row(d, x_trace, y_panel, TRACE_W, 50, demo["prox_ts"], t, n,
                           prox_vmax, f"prox Σ (a.u., max {prox_vmax:.2f})", onset_step, gate_step, f_small)
                _trace_row(d, x_trace, y_panel + 58, TRACE_W, 50, demo["force_ts"], t, n,
                           force_vmax, f"force Σ (N, max {force_vmax:.1f})", onset_step, gate_step, f_small)

        yield np.asarray(canvas)


def render_one(path: Path, demo: str, out: Path, scale: int, fps: float, quality: int,
               stride: int = 1, sensing: bool = True, markers: bool = False) -> Path:
    import imageio.v2 as imageio

    data = load_demo(path, demo)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out), fps=fps / max(stride, 1), codec="libx264", quality=quality,
        pixelformat="yuv420p", macro_block_size=None,
    )
    count = 0
    try:
        for frame in compose(data, path, demo, scale, fps, stride, sensing, markers):
            writer.append_data(frame)
            count += 1
    finally:
        writer.close()
    eff_fps = fps / max(stride, 1)
    print(f"{path.name}:{demo} -> {out}  ({count} frames @ {eff_fps:g} fps = {count / eff_fps:.1f}s"
          + (f", 1 of every {stride} steps" if stride > 1 else "") + ")")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", type=Path, help="dataset file written by scripts/generate.py")
    ap.add_argument("--demo", action="append", default=None, help="demo index or name (repeatable; default 0)")
    ap.add_argument("--out", type=Path, default=None, help="output .mp4 (single demo only)")
    ap.add_argument("--scale", type=int, default=3, help="nearest-neighbour upscale per camera (default 3 -> 384px)")
    ap.add_argument("--fps", type=float, default=20.0, help="playback rate (default 20 = policy rate = real time)")
    ap.add_argument("--quality", type=int, default=8, help="libx264 quality 0-10 (default 8)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame (smaller file, faster)")
    ap.add_argument("--no-sensing", action="store_true",
                    help="camera-only cut: no tactile / thermal field band")
    ap.add_argument("--markers", action="store_true",
                    help="overlay the marker lattice on the gel panel (dot = undeformed "
                         "marker, segment = where it moved); needs a marker-field file")
    args = ap.parse_args()

    if not args.hdf5.exists():
        print(f"ERROR: {args.hdf5} not found")
        return 2
    demos = args.demo or ["0"]
    demos = [d if str(d).startswith("demo_") else f"demo_{d}" for d in demos]
    if args.out is not None and len(demos) > 1:
        print("ERROR: --out takes a single --demo; drop one of them")
        return 2

    for d in demos:
        out = args.out or args.hdf5.with_name(f"{args.hdf5.stem}_{d}.mp4")
        try:
            render_one(args.hdf5, d, out, args.scale, args.fps, args.quality, args.every,
                       sensing=not args.no_sensing, markers=args.markers)
        except UnknownDemo as exc:
            print(f"ERROR: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
