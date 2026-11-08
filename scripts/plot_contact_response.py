"""画一份 episode 的温度/力响应曲线（时间 vs 指尖温度、法向力）。

用法::

    python scripts/plot_contact_response.py <episode.hdf5> --out curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.schema.thermal_contact_v1 import PHASE_NAMES  # noqa: E402
from data.writer.hdf5_writer import read_episode  # noqa: E402

# 中文字体的候选列表；系统里装了哪个就用哪个。
CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",
    "SimHei",
    "Microsoft YaHei",
)

LABELS_ZH = {
    "left": "左指",
    "right": "右指",
    "temp": "指尖温度 (°C)",
    "force": "法向力 (N)",
    "width": "夹爪开口 (mm)",
    "time": "时间 (s)",
    "title": "{id}  区域={region}  质量={quality}",
}
LABELS_EN = {
    "left": "left finger",
    "right": "right finger",
    "temp": "fingertip temperature (°C)",
    "force": "normal force (N)",
    "width": "gripper width (mm)",
    "time": "time (s)",
    "title": "{id}  region={region}  quality={quality}",
}


def setup_fonts() -> dict[str, str]:
    """有中文字体就用中文标注，否则退回英文（避免画出一堆方框）。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in installed:
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return LABELS_ZH
    return LABELS_EN


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="画温度/力响应曲线")
    parser.add_argument("path")
    parser.add_argument("--out", default=None, help="输出 PNG（默认与输入同名的 .png）")
    args = parser.parse_args()

    arrays, metadata, _ = read_episode(args.path)
    t = np.asarray(arrays["time/sim_time"], dtype=np.float64)
    temp = np.asarray(arrays["obs/fingertip_temp"], dtype=np.float64)
    normal = np.asarray(arrays["obs/contact_normal_force"], dtype=np.float64)
    width = np.asarray(arrays["obs/gripper_width"], dtype=np.float64).reshape(-1)
    phase = np.asarray(arrays["action/phase"]).reshape(-1)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    labels = setup_fonts()
    axes[0].plot(t, temp[:, 0], label=labels["left"])
    axes[0].plot(t, temp[:, 1], label=labels["right"])
    axes[0].set_ylabel(labels["temp"])
    axes[0].legend(loc="best")
    axes[1].plot(t, normal[:, 0], label=labels["left"])
    axes[1].plot(t, normal[:, 1], label=labels["right"])
    axes[1].set_ylabel(labels["force"])
    axes[1].legend(loc="best")
    axes[2].plot(t, width * 1e3, color="tab:gray")
    axes[2].set_ylabel(labels["width"])
    axes[2].set_xlabel(labels["time"])

    # 阶段切换线 + 名称：每条竖线标注新阶段。
    previous = None
    for i, code in enumerate(phase):
        if code == previous:
            continue
        previous = code
        for ax in axes:
            ax.axvline(t[i], color="tab:orange", lw=0.6, alpha=0.6)
        axes[0].annotate(
            PHASE_NAMES.get(int(code), str(code)),
            xy=(t[i], axes[0].get_ylim()[1]),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize=7,
            rotation=90,
            va="top",
            color="tab:orange",
        )

    region = metadata.get("target_region")
    title = labels["title"].format(
        id=metadata.get("episode_id"), region=region, quality=metadata.get("contact_quality")
    )
    fig.suptitle(title)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.path).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
