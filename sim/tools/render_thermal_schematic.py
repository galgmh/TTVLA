"""Render the four cup links as a temperature-colored schematic MP4.

The renderer reads only an existing episode HDF5. It does not change the
dataset and does not require Genesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REGIONS = ("upper", "middle", "lower", "handle")
REGION_COLUMNS = {"lower": 0, "middle": 1, "upper": 2, "handle": 3}


def _temperature_color(temp_c: float, lo: float, hi: float) -> tuple[int, int, int]:
    import cv2

    value = int(np.clip(round(255.0 * (float(temp_c) - lo) / (hi - lo)), 0, 255))
    pixel = np.asarray([[value]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.applyColorMap(pixel, cv2.COLORMAP_TURBO)[0, 0])


def _draw_centered(frame, text: str, center: tuple[int, int], *, scale=0.65, color=(255, 255, 255), thickness=2):
    import cv2

    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (int(center[0] - size[0] / 2), int(center[1] + size[1] / 2))
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _render_frame(
    temps: np.ndarray,
    time_s: float,
    contact_mask: np.ndarray,
    *,
    temp_min: float,
    temp_max: float,
) -> np.ndarray:
    import cv2

    height, width = 620, 900
    frame = np.full((height, width, 3), 244, dtype=np.uint8)
    ink = (38, 38, 38)
    muted = (105, 105, 105)

    cv2.putText(frame, "Cup link temperature", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.05, ink, 2, cv2.LINE_AA)
    cv2.putText(frame, f"t = {time_s:6.2f} s", (680, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.68, ink, 2, cv2.LINE_AA)

    body_left, body_right = 185, 540
    top, bottom = 105, 515
    boundaries = {
        "upper": (top, 235),
        "middle": (239, 372),
        "lower": (376, bottom),
    }
    centers: dict[str, tuple[int, int]] = {}
    for region in ("upper", "middle", "lower"):
        y0, y1 = boundaries[region]
        col = REGION_COLUMNS[region]
        color = _temperature_color(float(temps[col]), temp_min, temp_max)
        cv2.rectangle(frame, (body_left, y0), (body_right, y1), color, -1)
        touched = bool(contact_mask[:, col].any())
        cv2.rectangle(frame, (body_left, y0), (body_right, y1), (255, 255, 255) if touched else ink, 5 if touched else 2)
        centers[region] = ((body_left + body_right) // 2, (y0 + y1) // 2)

    # Handle is an open ring attached to the middle/upper body.
    handle_col = REGION_COLUMNS["handle"]
    handle_color = _temperature_color(float(temps[handle_col]), temp_min, temp_max)
    cv2.ellipse(frame, (570, 302), (150, 120), 0, -80, 80, handle_color, 50, cv2.LINE_AA)
    handle_touched = bool(contact_mask[:, handle_col].any())
    cv2.ellipse(frame, (570, 302), (150, 120), 0, -80, 80, (255, 255, 255) if handle_touched else ink, 5 if handle_touched else 2, cv2.LINE_AA)
    centers["handle"] = (690, 302)

    for region in REGIONS:
        col = REGION_COLUMNS[region]
        center = centers[region]
        _draw_centered(frame, region, (center[0], center[1] - 15), scale=0.68)
        _draw_centered(frame, f"{float(temps[col]):.2f} C", (center[0], center[1] + 20), scale=0.72)

    # Temperature legend.
    legend_x0, legend_x1 = 790, 825
    legend_y0, legend_y1 = 105, 515
    values = np.linspace(255, 0, legend_y1 - legend_y0 + 1, dtype=np.uint8).reshape(-1, 1)
    legend = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    legend = np.repeat(legend, legend_x1 - legend_x0 + 1, axis=1)
    frame[legend_y0 : legend_y1 + 1, legend_x0 : legend_x1 + 1] = legend
    cv2.rectangle(frame, (legend_x0, legend_y0), (legend_x1, legend_y1), ink, 1)
    for fraction in np.linspace(0.0, 1.0, 6):
        y = int(round(legend_y1 - fraction * (legend_y1 - legend_y0)))
        value = temp_min + fraction * (temp_max - temp_min)
        cv2.line(frame, (legend_x1 + 1, y), (legend_x1 + 8, y), ink, 1)
        cv2.putText(frame, f"{value:.0f}", (legend_x1 + 13, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ink, 1, cv2.LINE_AA)
    cv2.putText(frame, "C", (800, 548), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1, cv2.LINE_AA)

    if contact_mask.any():
        fingers = [name for i, name in enumerate(("L", "R")) if contact_mask[i].any()]
        cv2.putText(frame, f"Contact: {'+'.join(fingers)}", (42, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ink, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Contact: none", (42, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.65, muted, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a false-color four-link cup temperature MP4")
    parser.add_argument("episode", type=Path, help="Input episode HDF5")
    parser.add_argument("--out", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--temp-min", type=float, default=20.0, help="Color scale minimum in Celsius")
    parser.add_argument("--temp-max", type=float, default=70.0, help="Color scale maximum in Celsius")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS; defaults to control/time")
    args = parser.parse_args()
    if args.temp_max <= args.temp_min:
        raise ValueError("--temp-max must be greater than --temp-min")

    import cv2
    import h5py

    output = args.out or args.episode.with_name(f"{args.episode.stem}_thermal_schematic.mp4")
    with h5py.File(args.episode, "r") as handle:
        control_time = np.asarray(handle["control/time"], dtype=np.float64)
        sensor_index = np.asarray(handle["control/sensor_index"], dtype=np.int64)
        link_temp = np.asarray(handle["privileged/link_temp"], dtype=np.float64)
        contact = np.asarray(handle["privileged/contact_link_mask"], dtype=bool)
    if control_time.size == 0:
        raise ValueError("control/time is empty")
    fps = float(args.fps) if args.fps is not None else (
        1.0 / float(np.median(np.diff(control_time))) if control_time.size > 1 else 20.0
    )
    if fps <= 0.0:
        raise ValueError("FPS must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (900, 620))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create MP4: {output}")
    try:
        for t, index in zip(control_time, sensor_index):
            writer.write(
                _render_frame(
                    link_temp[index], float(t), contact[index],
                    temp_min=float(args.temp_min), temp_max=float(args.temp_max),
                )
            )
    finally:
        writer.release()
    print(f"FPS: {fps:.3f}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
