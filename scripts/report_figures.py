#!/usr/bin/env python
"""Regenerate the status-report figures under docs/figures/.

    python scripts/report_figures.py

Four figures (matplotlib only, no genesis import — a dataloader-tier script):

  1. thermal_channel.png     -- the temperature channel over episodes (data/)
  2. tactile_field_and_map.png -- 63-point marker field vs the height map synthesized
                                  from it (format 1.3 preview run, /tmp/hm3)
  3. kernel_grid_coupling.png -- R25: same grasp on a 16x16 vs a 9x7 marker grid
                                  (the elastic kernel widens with the grid spacing)
  4. milestones.png          -- milestone status + capability-readiness estimate

The /tmp inputs are the format-1.3 preview runs; pass --pair-h5/--ab-256/--ab-63
to point at other files, or the corresponding figure is skipped with a warning.
See docs/status.md for what each figure shows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempvla.tools import height_map as hm  # noqa: E402
from tempvla.tools import tactile_field as tf  # noqa: E402

import h5py  # noqa: E402

MATERIALS = ["copper", "aluminum", "steel", "abs"]
MAT_COLORS = {"copper": "#d35400", "aluminum": "#7f8c8d", "steel": "#2c3e50", "abs": "#27ae60"}
DONE, PARTIAL, TODO = "#27ae60", "#f39c12", "#95a5a6"


# --------------------------------------------------------------------- helpers


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )
    return plt


def _grid_pitch_mm(positions: np.ndarray, grid: tuple[int, int]) -> tuple[float, float]:
    """(dx, dy) marker pitch in mm from the stored probe positions (linear span)."""
    ny, nx = grid
    p = np.asarray(positions, dtype=np.float64).reshape(ny, nx, 3)
    dy = float(np.linalg.norm(p[-1].mean(axis=0) - p[0].mean(axis=0))) / (ny - 1)
    dx = float(np.linalg.norm(p[:, -1].mean(axis=0) - p[:, 0].mean(axis=0))) / (nx - 1)
    return dx * 1e3, dy * 1e3


def _peak_um(plane: np.ndarray) -> np.ndarray:
    """(T, F, ny, nx) metres -> (F,) peak |pad-normal displacement| in um."""
    return np.abs(plane).max(axis=(0, 2, 3)) * 1e6


def _lattice_index_coords(
    pos_src: np.ndarray, grid_src: tuple[int, int], pos_dst: np.ndarray, grid_dst: tuple[int, int]
) -> np.ndarray:
    """Map ``pos_dst`` points into the ``pos_src`` lattice index space.

    Same linear-span model the renderer uses: the lattice tangent axes are the
    mean column/row directions, and each point is projected onto them.
    Returns (F, 2, n_dst) with rows = (row index, col index).
    """
    out = []
    for f in range(np.asarray(pos_dst).shape[0]):
        p = np.asarray(pos_src)[f].reshape(*grid_src, 3)
        origin = p[0, 0]
        across = p[:, -1].mean(0) - p[:, 0].mean(0)
        along = p[-1].mean(0) - p[0].mean(0)
        basis = np.stack([across, along], 1)  # 3x2
        t = np.asarray(pos_dst)[f].reshape(*grid_dst, 3) - origin
        ab = np.linalg.lstsq(basis, t.reshape(-1, 3).T, rcond=None)[0]
        out.append(np.stack([ab[1] * (grid_src[0] - 1), ab[0] * (grid_src[1] - 1)]))
    return np.array(out)


# ------------------------------------------------------------------ figure 1


def fig_thermal(data_dir: Path, out: Path) -> bool:
    """Temperature channel: contact-gated rise, object-temp dependence, locality."""
    plt = _pyplot()
    curves, finals = {}, {}
    for mat in MATERIALS:
        path = data_dir / f"pick_hot_cylinder_{mat}.hdf5"
        if not path.exists():
            print(f"  [thermal] missing {path} — skipped")
            return False
        rise, t0 = [], []
        with h5py.File(path, "r") as f:
            tmin = min(f[f"data/{d}/obs/temperature/fingertip_grid"].shape[0] for d in f["data"])
            for d in f["data"]:
                ft = f[f"data/{d}/obs/temperature/fingertip_grid"][:tmin, :, :, :, 0]
                ob = f[f"data/{d}/obs/temperature/object"][:]
                curves.setdefault(mat, []).append(ft.mean(axis=(1, 2, 3)) - ft[0].mean())
                rise.append(float(ft[-1].mean() - ft[0].mean()))
                t0.append(float(ob[0, 0]))
        finals[mat] = (np.array(rise), np.array(t0))

    first_contact = []
    with h5py.File(data_dir / "pick_hot_cylinder_aluminum.hdf5", "r") as f:
        for d in f["data"]:
            lc = f[f"data/{d}/obs/contact/link_contact"][:]
            if lc.any():
                first_contact.append(int(np.argmax(lc.any(axis=(1, 2)))))
    contact_step = int(np.median(first_contact)) if first_contact else 0

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axs[0]
    for mat in MATERIALS:
        c = np.median(np.stack(curves[mat]), axis=0)
        ax.plot(np.arange(c.size), c, color=MAT_COLORS[mat], lw=1.8, label=mat)
    ax.axvline(contact_step, color="k", ls="--", lw=1)
    ax.annotate("contact", (contact_step, ax.get_ylim()[1] * 0.05), xytext=(contact_step + 1.2, 1), fontsize=8.5)
    ax.set_xlabel("policy step (20 Hz)")
    ax.set_ylabel(r"$\Delta T$ fingertip (mean of both, °C)")
    ax.set_title("(a) heating starts only at contact\n(adiabatic: conv/rad disabled, R14)")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axs[1]
    for mat in MATERIALS:
        rise, t0 = finals[mat]
        ax.scatter(t0, rise, s=14, alpha=0.65, color=MAT_COLORS[mat], label=mat, edgecolors="none")
    allr = np.concatenate([finals[m][0] for m in MATERIALS])
    allt = np.concatenate([finals[m][1] for m in MATERIALS])
    r = float(np.corrcoef(allt, allr)[0, 1])
    k, b = np.polyfit(allt, allr, 1)
    xs = np.linspace(allt.min(), allt.max(), 10)
    ax.plot(xs, k * xs + b, color="k", lw=1, ls=":")
    ax.text(0.05, 0.92, f"Pearson r = {r:.2f}  (n = {allr.size})", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("object initial temperature (°C)")
    ax.set_ylabel(r"$\Delta T$ fingertip over episode (°C)")
    ax.set_title("(b) the channel tracks the object's temperature\n(absolute scale uncalibrated, R16)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axs[2]
    rise, t0 = finals["copper"]
    pick = int(np.argmin(np.abs(rise - np.median(rise))))
    with h5py.File(data_dir / "pick_hot_cylinder_copper.hdf5", "r") as f:
        d = list(f["data"])[pick]
        g = f[f"data/{d}/obs/temperature/fingertip_grid"][-1, :, :, :, 0]  # (2,4,4)
        t0r = float(f[f"data/{d}/obs/temperature/object"][0, 0])
    vmax = float(g.max())
    im = ax.imshow(np.concatenate([g[0], np.full((4, 1), np.nan), g[1]], axis=1), cmap="inferno", vmin=21.0, vmax=vmax)
    for i in range(2):
        for rr in range(4):
            for cc in range(4):
                ax.text(i * 5 + cc, rr, f"{g[i, rr, cc]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="w" if g[i, rr, cc] < 0.6 * vmax else "k")
    ax.set_xticks([1.5, 6.5], ["tip L", "tip R"]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"(c) copper {d}, final 4x4 fingertip grids (°C)\nheat enters only the contact band (T0 = {t0r:.0f} °C)")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="°C")

    fig.suptitle(
        "Temperature channel — pick_hot_cylinder (v1 dataset, 4 materials x 50 episodes; the object barely cools, the finger absorbs)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "thermal_channel.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [thermal] wrote {path}  (contact step {contact_step}, r={r:.2f})")
    return True


# ------------------------------------------------------------------ figure 2


def fig_field_and_map(pair_h5: Path, out: Path) -> bool:
    """63-point marker field vs the synthesized 64x64 height map, box vs cylinder."""
    plt = _pyplot()
    if not pair_h5.exists():
        print(f"  [field/map] missing {pair_h5} — skipped")
        return False
    with h5py.File(pair_h5, "r") as f:
        grid = tf.modality_grid(f, hm.MARKER_MODALITY)
        pos, _ = tf.finger_geometry(f, hm.MARKER_MODALITY)
        dx, dy = _grid_pitch_mm(pos[0], grid)
        ny, nx = grid
        extent = (0.0, nx * dx, 0.0, ny * dy)
        rows = []
        for d in ("demo_0", "demo_1"):
            plane = tf.pad_normal_field(f, d, hm.MARKER_MODALITY)  # (T, F, ny, nx) m
            hmap = hm.height_map_or_synth(f, d)  # (T, F, H, W) m
            shape = str(f[f"data/{d}"].attrs["object_shape"])
            amp = np.abs(plane).max(axis=(1, 2, 3))  # (T,)
            t = int(np.argmax(amp))
            fsel = int(np.argmax(np.abs(plane[t]).max(axis=(1, 2))))
            peak = np.abs(plane).max() * 1e6
            mask = np.abs(plane[t, fsel]) > 0.5 * np.abs(plane).max()
            across = int(np.count_nonzero(mask.any(axis=0)))
            along = int(np.count_nonzero(mask.any(axis=1)))
            rows.append((d, shape, plane[t, fsel] * 1e6, hmap[t, fsel] * 1e6, peak, across, along, t))

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.2))
    vmax = float(np.ceil(max(r_[4] for r_ in rows) / 2.0) * 2.0)
    im = None
    for i, (d, shape, field, hmap, peak, across, along, t) in enumerate(rows):
        for j, (img, tag) in enumerate(((field, f"{ny}x{nx} marker field — {nx * ny} samples"), (hmap, f"{hmap.shape[0]}x{hmap.shape[1]} height map — synthesized"))):
            ax = axs[i, j]
            im = ax.imshow(img, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax,
                           interpolation="nearest" if j == 0 else "bilinear", aspect="equal")
            xs = (np.arange(nx) + 0.5) * dx
            ys = (np.arange(ny) + 0.5) * dy
            ax.plot(*np.meshgrid(xs, ys, indexing="xy"), ls="none", marker="+", ms=3.5, mew=0.6, color="cyan", alpha=0.8)
            ax.set_title(f"{shape} ({d}) — {tag}\nframe {t}, peak {peak:.2f} µm", fontsize=9)
            ax.set_xlabel("pad cols (mm)")
            ax.set_ylabel("pad rows (mm)")
            ax.grid(False)
    # imprint extent annotation from the field row
    ann = ", ".join(f"{r_[1]} {r_[5]}/{nx} cols x {r_[6]}/{ny} rows" for r_ in rows)
    fig.suptitle(
        f"63-point marker field vs the height map synthesized from it (shared absolute scale)\n"
        f"> 50% of peak: {ann}\n"
        f"pad patch {nx * dx:.1f} x {ny * dy:.1f} mm — pitch {dx:.2f} mm (cols) / {dy:.2f} mm (rows); cyan + = marker nodes",
        fontsize=10,
    )
    fig.colorbar(im, ax=list(axs.ravel()), fraction=0.04, pad=0.02, label="|pad-normal displacement| (µm)")
    fig.savefig(out / "tactile_field_and_map.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [field/map] wrote {out / 'tactile_field_and_map.png'}  ({ann})")
    return True


# ------------------------------------------------------------------ figure 3


def fig_kernel_coupling(ab256: Path, ab63: Path, out: Path) -> bool:
    """R25: the same grasp on a 16x16 vs a 9x7 marker grid (kernel width in cells)."""
    plt = _pyplot()
    from scipy.ndimage import map_coordinates

    if not (ab256.exists() and ab63.exists()):
        print(f"  [kernel] missing {ab256} or {ab63} — skipped")
        return False
    data = {}
    for tag, path in (("16", ab256), ("9", ab63)):
        with h5py.File(path, "r") as f:
            grid = tf.modality_grid(f, hm.MARKER_MODALITY)
            pos, _ = tf.finger_geometry(f, hm.MARKER_MODALITY)
            planes = {d: tf.pad_normal_field(f, d, hm.MARKER_MODALITY) for d in f["data"]}
            data[tag] = (grid, np.asarray(pos), planes)
    g256, pos256, pl256 = data["16"]
    g63, pos63, pl63 = data["9"]

    # per (demo, finger): measured peaks + the 16x16 field resampled at the 63 positions
    peaks = {}
    coords = _lattice_index_coords(pos256, g256, pos63, g63)  # (F, 2, 63)
    for d in pl256:
        pred = np.zeros((2,), dtype=float)
        for f in range(2):
            vals = np.stack(
                [map_coordinates(pl256[d][t, f], coords[f], order=3, mode="nearest") for t in range(pl256[d].shape[0])]
            )
            pred[f] = np.abs(vals).max() * 1e6
        peaks[d] = (_peak_um(pl256[d]), pred, _peak_um(pl63[d]))

    dx9, dy9 = _grid_pitch_mm(pos63[0], g63)
    dx16, dy16 = _grid_pitch_mm(pos256[0], g256)

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0), hspace=0.42, wspace=0.3)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axb = fig.add_subplot(gs[1, :])

    d, fsel = "demo_0", 0
    plane = pl256[d] * 1e6
    t = int(np.argmax(np.abs(plane).max(axis=(1, 2, 3))))
    a = plane[t, fsel]
    pred_map = map_coordinates(plane[t, fsel], coords[fsel], order=3, mode="nearest").reshape(g63[0], g63[1])
    c = pl63[d][t, fsel] * 1e6

    vmax = float(a.max()) * 1.08
    panels = (
        (a, g256, dx16, dy16, f"(a) 16x16 measured — {g256[0] * g256[1]} samples\npeak {_peak_um(pl256[d])[fsel]:.1f} µm"),
        (pred_map, g63, dx9, dy9, f"(b) the same 16x16 field, read only at the\n63 positions — prediction {np.abs(pred_map).max():.1f} µm"),
        (c, g63, dx9, dy9, f"(c) 9x7 measured, same grasp/seed\npeak {_peak_um(pl63[d])[fsel]:.1f} µm"),
    )
    im = None
    for ax, (img, grid, dx, dy, title) in zip(axs, panels):
        ny, nx = grid
        im = ax.imshow(img, origin="lower", extent=(0, nx * dx, 0, ny * dy), cmap="magma",
                       vmin=0.0, vmax=vmax, interpolation="bilinear", aspect="equal")
        xs, ys = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
        ax.plot(*np.meshgrid(xs, ys, indexing="xy"), ls="none", marker="+", ms=3, mew=0.5, color="cyan", alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("pad cols (mm)")
        ax.grid(False)
    axs[0].set_ylabel("pad rows (mm)")
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.015, label="|pad-normal displacement| (µm)")

    labels = ["cyl. tip L", "cyl. tip R", "box tip L", "box tip R"]
    groups = [("demo_0", 0), ("demo_0", 1), ("demo_1", 0), ("demo_1", 1)]
    x = np.arange(len(groups))
    w = 0.26
    vals = np.array([[peaks[d][0][f], peaks[d][1][f], peaks[d][2][f]] for d, f in groups])
    bars = (
        axb.bar(x - w, vals[:, 0], w, label="16x16 measured", color="#2c3e50"),
        axb.bar(x, vals[:, 1], w, label="9x7 predicted (resampled 16x16)", color="#5dade2"),
        axb.bar(x + w, vals[:, 2], w, label="9x7 measured", color="#d35400"),
    )
    for xi, row in zip(x, vals):
        for off, v in zip((-w, 0, w), row):
            axb.text(xi + off, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
        axb.text(xi + w, row[2] + 2.6, f"{row[2] / row[1] * 100:.0f}%", ha="center", fontsize=8, color="#d35400")
    axb.set_xticks(x, labels)
    axb.set_ylabel("peak |pad-normal displacement| (µm)")
    axb.set_title(
        "peak per grasp (finger 0/1): the 2.5x coarser grid widens the elastic kernel (it is applied in FFT cells), "
        "so the peak keeps only ~22% of the prediction;\nresampling alone (16x16 read at the 63 positions) would keep 96–100% — "
        "the loss is the kernel, not the sampling", fontsize=9)
    axb.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    axb.set_ylim(0, vmax * 1.35)

    fig.suptitle("R25 — same grasp, two marker grids: elastic kernel width scales with grid spacing (format 1.3 preview run)", fontsize=11)
    fig.savefig(out / "kernel_grid_coupling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [kernel] wrote {out / 'kernel_grid_coupling.png'}  (peaks {peaks})")
    return True


# ------------------------------------------------------------------ figure 4

MILESTONES = [
    ("M0", "environment + Gates 0-2", 1.0, DONE, "done"),
    ("M1", "minimal episode -> HDF5", 1.0, DONE, "done"),
    ("M2", "v1 dataset: 4 materials x 50 (200/200)", 1.0, DONE, "done"),
    ("M3", "cameras: Gate 3/4, 256^2 (v2 dataset = 128^2)", 1.0, DONE, "done"),
    ("M4", "3-task suite + replay + data card", 0.4, PARTIAL, "partial (2 of 3 tasks)"),
    ("M5", "format 1.1: marker field", 1.0, DONE, "done"),
    ("M6", "format 1.2: two-object task", 1.0, DONE, "done"),
    ("M7", "format 1.3: 63 markers + height map", 1.0, DONE, "done (preview only)"),
    ("M8", "regenerate full datasets on 1.3", 0.0, TODO, "not started"),
    ("M9", "fusion baseline (temp + tactile)", 0.0, TODO, "not started"),
    ("M10", "tactile sim-to-real (gsmini)", 0.0, TODO, "not started"),
    ("M11", "thermal sim-to-real", 0.0, TODO, "not started"),
    ("M12", "real-arm transfer", 0.0, TODO, "not started"),
]

GAPS = [
    ("thermal fidelity", 0.25, "causal + local, but adiabatic & gain uncalibrated (R14/R16)"),
    ("fusion VLA baseline", 0.00, "no policy trained on the channels yet"),
    ("sim-to-real alignment", 0.05, "no real sensor data, no hardware"),
    ("tactile fidelity", 0.40, "field structure right; amplitude/kernel uncalibrated (R25)"),
    ("data pipeline (sim)", 0.85, "multi-object + cameras + 1.3 stack; full set not regenerated"),
]


def fig_milestones(out: Path) -> bool:
    plt = _pyplot()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14.0, 5.6), width_ratios=(1.5, 1.0))

    labels = [f"{m}  {name}" for m, name, *_ in MILESTONES][::-1]
    vals = [v for *_, v, _c, _s in MILESTONES][::-1]
    cols = [c for *_, c, _s in MILESTONES][::-1]
    notes = [s for *_, s in MILESTONES][::-1]
    y = np.arange(len(labels))
    axl.barh(y, [1.0] * len(y), color="#ecf0f1", height=0.62)
    axl.barh(y, vals, color=cols, height=0.62)
    for yi, (v, note) in enumerate(zip(vals, notes)):
        axl.text(max(v, 0.02) + 0.02, yi, note, va="center", fontsize=8, color="#444")
    axl.set_yticks(y, labels, fontsize=8.5)
    axl.set_xlim(0, 1.85)
    axl.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    axl.set_title("milestones (M0-M7 shipped & verified; M4 partial; M8+ proposed here)", fontsize=10)
    axl.grid(axis="y", visible=False)

    labels = [g[0] for g in GAPS][::-1]
    vals = [g[1] for g in GAPS][::-1]
    notes = [g[2] for g in GAPS][::-1]
    y = np.arange(len(labels))
    axr.barh(y, [1.0] * len(y), color="#ecf0f1", height=0.62)
    bars = axr.barh(y, vals, color=[plt.cm.RdYlGn(v) for v in vals], height=0.62)
    for yi, (v, note, b) in enumerate(zip(vals, notes, bars)):
        axr.text(1.02, yi, f"{v * 100:.0f}%", va="center", fontsize=8.5)
        axr.text(0.01, yi - 0.42, note, va="center", fontsize=7.5, color="#555")
    axr.set_yticks(y, labels, fontsize=9)
    axr.set_xlim(0, 1.12)
    axr.set_ylim(-0.75, len(labels) - 0.25)
    axr.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    axr.set_title("capability readiness (subjective estimate)", fontsize=10)
    axr.grid(axis="y", visible=False)

    fig.suptitle(
        "Goal: temperature + contact-tactile VLA data that transfers to a real arm — where the project stands (2026-10-08)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "milestones.png")
    plt.close(fig)
    print(f"  [milestones] wrote {out / 'milestones.png'}")
    return True


# ----------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="v1 dataset directory (figure 1)")
    parser.add_argument("--pair-h5", type=Path, default=Path("/tmp/hm3/lift_object_pair_aluminum.hdf5"),
                        help="format-1.3 pair preview run with the height map (figure 2)")
    parser.add_argument("--ab-256", type=Path, default=Path("/tmp/m256/lift_object_pair_aluminum.hdf5"),
                        help="16x16 A/B run (figure 3)")
    parser.add_argument("--ab-63", type=Path, default=Path("/tmp/m63/lift_object_pair_aluminum.hdf5"),
                        help="9x7 A/B run (figure 3)")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"), help="output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ok = [
        fig_thermal(args.data_dir, args.out),
        fig_field_and_map(args.pair_h5, args.out),
        fig_kernel_coupling(args.ab_256, args.ab_63, args.out),
        fig_milestones(args.out),
    ]
    print(f"{sum(ok)}/{len(ok)} figures written to {args.out}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
