#!/usr/bin/env python
"""Check a LeRobot v2.1 dataset written by ``experiment/export_lerobot.py``.

Runs without Genesis and without ``lerobot``: it reads the metadata, the
parquet files (pyarrow) and the MP4 paths, and checks them against each other.
With ``--decode`` it also opens each video with imageio and checks the frame
count, size and fps. With ``--with-lerobot`` it additionally loads the dataset
through the installed ``lerobot`` package, which is the strongest check that
openpi's loader will accept it.

    python experiment/check_lerobot.py datasets/lerobot/local/pick_cold_can_p16
    python experiment/check_lerobot.py datasets/lerobot/local/pick_cold_can_p16 --decode --with-lerobot

Exit status is 1 if any issue was found (warnings do not fail the run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiment.common import STATE_NAMES, state_dim  # noqa: E402


# -------------------------------------------------------------------- report


class Report:
    def __init__(self) -> None:
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.passed = 0

    def ok(self, message: str) -> None:
        self.passed += 1

    def fail(self, message: str) -> None:
        self.issues.append(message)
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[warn] {message}")

    def check(self, cond: bool, message: str) -> bool:
        if cond:
            self.ok(message)
        else:
            self.fail(message)
        return bool(cond)


# ------------------------------------------------------------------ reading


def _json_file(path: Path, report: Report) -> Any | None:
    if not path.is_file():
        report.fail(f"missing {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.fail(f"cannot read {path}: {exc}")
        return None


def _jsonl_file(path: Path, report: Report) -> list[dict[str, Any]] | None:
    if not path.is_file():
        report.fail(f"missing {path}")
        return None
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        report.fail(f"cannot read {path}: {exc}")
        return None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            report.fail(f"{path.name}:{line_no}: bad JSON ({exc})")
            continue
        if not isinstance(row, dict):
            report.fail(f"{path.name}:{line_no}: row is not an object")
            continue
        rows.append(row)
    return rows


# ------------------------------------------------------------------- checks


def _check_info(info: dict[str, Any], report: Report) -> dict[str, Any]:
    features = info.get("features")
    if not report.check(isinstance(features, dict) and bool(features), "features block present"):
        return {}
    report.check(info.get("codebase_version") == "v2.1", "codebase_version is v2.1")
    report.check(isinstance(info.get("robot_type"), str), "robot_type present")
    for key in ("total_episodes", "total_frames", "total_tasks", "total_videos", "chunks_size"):
        value = info.get(key)
        if not report.check(isinstance(value, int) and value >= 0, f"{key} is a non-negative int"):
            continue
    fps = info.get("fps")
    report.check(isinstance(fps, (int, float)) and float(fps) > 0, "fps > 0")

    for key in (
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.state",
        "action",
    ):
        report.check(key in features, f"feature {key!r} declared")
    state = features.get("observation.state") or {}
    action = features.get("action") or {}
    state_shape = state.get("shape")
    dim = state_shape[0] if isinstance(state_shape, list) and state_shape else None
    report.check(isinstance(dim, int) and dim > 0, "observation.state has a positive width")
    names = state.get("names")
    report.check(
        isinstance(names, list) and len(names) == dim,
        "observation.state names match its width",
    )
    report.check(
        action.get("shape") == [7],
        f"action is 7-dim (got {action.get('shape')!r})",
    )
    report.check(
        isinstance(action.get("names"), list) and len(action["names"]) == 7,
        "action names match 7 dims",
    )

    videos = {k: v for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"}
    report.check(bool(videos), "at least one video feature declared")
    for key, spec in videos.items():
        info_block = spec.get("info") or {}
        report.check(
            float(info_block.get("video.fps", -1)) == float(fps or -2),
            f"{key}: video fps matches info fps",
        )
        shape = spec.get("shape") or []
        report.check(
            len(shape) == 3 and shape[2] == 3 and shape[0] == shape[1],
            f"{key}: square RGB shape",
        )
        report.check(int(shape[0] or 0) % 2 == 0, f"{key}: even frame size")
    report.check(
        isinstance(info.get("data_path"), str) and "{episode_index" in info["data_path"],
        "data_path template present",
    )
    report.check(
        isinstance(info.get("video_path"), str) and "{video_key" in info["video_path"],
        "video_path template present",
    )
    return features


def _check_tasks(rows: list[dict[str, Any]], info: dict[str, Any], report: Report) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for i, row in enumerate(rows):
        idx = row.get("task_index")
        text = row.get("task")
        if not report.check(isinstance(idx, int) and idx >= 0, f"tasks.jsonl row {i}: task_index"):
            continue
        if not report.check(isinstance(text, str) and bool(text.strip()), f"tasks.jsonl row {i}: task"):
            continue
        report.check(idx not in tasks, f"tasks.jsonl row {i}: task_index {idx} unique")
        tasks[idx] = text
    report.check(
        sorted(tasks) == list(range(len(tasks))),
        "task indices are contiguous from 0",
    )
    if isinstance(info.get("total_tasks"), int):
        report.check(len(tasks) == info["total_tasks"], "total_tasks matches tasks.jsonl")
    report.check(len(tasks) > 0, "at least one language instruction")
    lengths = [len(text.split()) for text in tasks.values()]
    if lengths and max(lengths) > 40:
        report.warn(f"instruction lengths: min {min(lengths)} / max {max(lengths)} tokens")
    else:
        report.ok("instruction lengths look short")
    return tasks


def _check_episodes_meta(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
    tasks: dict[int, str],
    report: Report,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: int(r.get("episode_index", -1)))
    report.check(
        [r.get("episode_index") for r in ordered] == list(range(len(ordered))),
        "episodes.jsonl indices are 0..N-1 in order",
    )
    if isinstance(info.get("total_episodes"), int):
        report.check(len(ordered) == info["total_episodes"], "total_episodes matches episodes.jsonl")
    total = 0
    for row in ordered:
        idx = row.get("episode_index")
        length = row.get("length")
        if not report.check(isinstance(length, int) and length > 0, f"episode {idx}: length > 0"):
            continue
        total += length
        task_list = row.get("tasks")
        if report.check(
            isinstance(task_list, list) and len(task_list) == 1 and isinstance(task_list[0], str),
            f"episode {idx}: exactly one task",
        ):
            report.check(task_list[0] in tasks.values(), f"episode {idx}: task is known")
    if isinstance(info.get("total_frames"), int):
        report.check(total == info["total_frames"], "total_frames matches the episode lengths")
    return ordered


def _check_parquet(
    root: Path,
    ep: dict[str, Any],
    info: dict[str, Any],
    tasks: dict[int, str],
    *,
    expected_index0: int,
    report: Report,
) -> int:
    idx = ep["episode_index"]
    length = ep["length"]
    chunk = idx // int(info["chunks_size"])
    path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
    if not report.check(path.is_file(), f"episode {idx}: parquet exists"):
        return expected_index0

    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.fail("pyarrow is required to inspect the parquet files (pip install pyarrow)")
        return expected_index0

    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - pyarrow raises many types here
        report.fail(f"episode {idx}: unreadable parquet ({exc})")
        return expected_index0

    expected_cols = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    actual = set(table.column_names)
    if actual != expected_cols:
        report.fail(
            f"episode {idx}: columns differ "
            f"(missing {sorted(expected_cols - actual)}, extra {sorted(actual - expected_cols)})"
        )
        return expected_index0
    report.ok(f"episode {idx}: columns match")

    n = table.num_rows
    report.check(n == length, f"episode {idx}: {n} rows == length {length}")

    task_col = set(table["task_index"].to_pylist())
    if report.check(len(task_col) == 1, f"episode {idx}: one task_index"):
        task_idx = int(next(iter(task_col)))
        ep_tasks = ep.get("tasks") or []
        report.check(
            bool(ep_tasks) and tasks.get(task_idx) == ep_tasks[0],
            f"episode {idx}: task_index {task_idx} names the same instruction",
        )
    report.check(
        set(table["episode_index"].to_pylist()) == {idx},
        f"episode {idx}: episode_index column is constant",
    )
    frame_index = table["frame_index"].to_pylist()
    report.check(frame_index == list(range(length)), f"episode {idx}: frame_index is 0..T-1")
    index = table["index"].to_pylist()
    report.check(
        index == list(range(expected_index0, expected_index0 + length)),
        f"episode {idx}: global index continues from {expected_index0}",
    )
    fps = float(info["fps"])
    stamps = table["timestamp"].to_pylist()
    want = [i / fps for i in range(length)]
    report.check(
        all(abs(float(a) - b) <= 1e-4 for a, b in zip(stamps, want)),
        f"episode {idx}: timestamp == frame_index / fps",
    )

    import numpy as np

    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    dim = state_dim_from_info(info)
    report.check(state.shape == (length, dim), f"episode {idx}: state shape (T,{dim})")
    report.check(action.shape == (length, 7), f"episode {idx}: action shape (T,7)")
    if state.ndim == 2 and state.size:
        report.check(bool(np.isfinite(state).all()), f"episode {idx}: state finite")
        report.check(bool(np.isfinite(action).all()), f"episode {idx}: action finite")
        if dim >= 7:
            gripper = state[:, 6]
            report.check(
                bool(((gripper >= -1.001) & (gripper <= 1.001)).all()),
                f"episode {idx}: gripper in [-1,1]",
            )
    return expected_index0 + n


def state_dim_from_info(info: dict[str, Any]) -> int:
    state = (info.get("features") or {}).get("observation.state") or {}
    shape = state.get("shape")
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return int(shape[0])
    return -1


def _check_videos(
    root: Path,
    episodes: list[dict[str, Any]],
    info: dict[str, Any],
    *,
    decode: bool,
    report: Report,
) -> None:
    videos = [
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    if not videos:
        return
    if isinstance(info.get("total_videos"), int):
        report.check(
            info["total_videos"] == len(episodes) * len(videos),
            "total_videos matches episodes x cameras",
        )

    imageio = None
    if decode:
        try:
            import imageio  # noqa: F401
        except ImportError:
            report.warn("--decode needs imageio/imageio-ffmpeg; skipping frame checks")
            decode = False

    fps = float(info.get("fps") or 0.0)
    for key in videos:
        missing = 0
        for ep in episodes:
            idx = ep["episode_index"]
            chunk = idx // int(info["chunks_size"])
            path = root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{idx:06d}.mp4"
            if not path.is_file():
                missing += 1
                continue
            if decode:
                frames, _meta = _decode_check(path, fps, report, key, idx)
                if frames is not None and frames != ep["length"]:
                    report.fail(
                        f"{key} episode {idx}: {frames} decoded frames != length {ep['length']}"
                    )
        report.check(missing == 0, f"{key}: every episode has an MP4 ({missing} missing)")


def _decode_check(
    path: Path,
    fps: float,
    report: Report,
    key: str,
    idx: int,
) -> tuple[int | None, dict[str, Any] | None]:
    import imageio

    try:
        reader = imageio.get_reader(str(path))
    except Exception as exc:  # noqa: BLE001
        report.fail(f"{key} episode {idx}: cannot open MP4 ({exc})")
        return None, None
    try:
        meta = reader.get_meta_data()
        try:
            frames = int(reader.count_frames())
        except Exception:  # noqa: BLE001 - codec dependent
            duration = float(meta.get("duration") or 0.0)
            frames = int(round(duration * fps))
        first = reader.get_data(0)
        mid = reader.get_data(max(0, frames // 2))
    except Exception as exc:  # noqa: BLE001 - codec dependent
        report.fail(f"{key} episode {idx}: cannot read frames ({exc})")
        return None, None
    finally:
        reader.close()
    report.check(
        first.shape == mid.shape and first.ndim == 3 and first.shape[2] == 3,
        f"{key} episode {idx}: RGB frames decode",
    )
    got_fps = float(meta.get("fps") or -1)
    report.check(abs(got_fps - fps) <= 0.51, f"{key} episode {idx}: fps {got_fps:g} ~ {fps:g}")
    return frames, meta


def _scalar_stat_values(blocks: dict[str, Any]) -> list[str]:
    """Return ``feature.stat`` paths whose value is not a list.

    LeRobot loads every stat as ``np.array(value)`` and its loader requires
    ``ndim >= 1``; a bare JSON number becomes a 0-d array and ``aggregate_stats``
    raises "Number of dimensions must be at least 1, and is 0 instead.".
    """
    offenders: list[str] = []
    for key, block in blocks.items():
        if not isinstance(block, dict):
            continue
        for field, value in block.items():
            if not isinstance(value, list):
                offenders.append(f"{key}.{field}")
    return offenders


def _list_issue(report: Report, label: str, offenders: list[str]) -> None:
    detail = f" (scalar: {', '.join(offenders[:3])})" if offenders else ""
    report.check(not offenders, f"{label}: every stat value is a list{detail}")


def _is_3x1x1(value: Any) -> bool:
    """LeRobot's image stats are one value per channel with two singleton axes."""
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(channel, list)
            and len(channel) == 1
            and isinstance(channel[0], list)
            and len(channel[0]) == 1
            for channel in value
        )
    )


def _check_stats(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    report: Report,
) -> None:
    stats = _json_file(root / "meta" / "stats.json", report)
    if not isinstance(stats, dict):
        return
    _list_issue(report, "stats.json", _scalar_stat_values(stats))
    dim = state_dim_from_info(info)
    total_frames = int(info.get("total_frames", 0))
    for key, width in (("observation.state", dim), ("action", 7)):
        block = stats.get(key)
        if not report.check(isinstance(block, dict), f"stats.json has {key}"):
            continue
        for field in ("mean", "std", "min", "max"):
            values = block.get(field)
            report.check(
                isinstance(values, list) and len(values) == width,
                f"stats.json {key}.{field} has {width} values",
            )
        if isinstance(block.get("mean"), list) and isinstance(block.get("min"), list):
            lo, mean, hi = block["min"], block["mean"], block["max"]
            if len(lo) == len(mean) == len(hi):
                report.check(
                    all(float(a) - 1e-3 <= float(b) <= float(c) + 1e-3 for a, b, c in zip(lo, mean, hi)),
                    f"stats.json {key}: min <= mean <= max",
                )
        report.check(
            block.get("count") == [total_frames],
            f"stats.json {key}.count == [total_frames]",
        )
    for key, spec in (info.get("features") or {}).items():
        if not (isinstance(spec, dict) and spec.get("dtype") == "video"):
            continue
        block = stats.get(key) or {}
        report.check(
            all(_is_3x1x1(block.get(field)) for field in ("mean", "std", "min", "max")),
            f"stats.json {key}: image stats are (3,1,1)",
        )
        report.check(
            block.get("count") == [total_frames],
            f"stats.json {key}.count == [total_frames]",
        )

    rows = _jsonl_file(root / "meta" / "episodes_stats.jsonl", report)
    if rows is None:
        return
    report.check(len(rows) == len(episodes), "episodes_stats.jsonl covers every episode")
    offenders = [
        f"{row.get('episode_index')}.{path}"
        for row in rows
        for path in _scalar_stat_values(row.get("stats") or {})
    ]
    _list_issue(report, "episodes_stats.jsonl", offenders)
    image_keys = [
        key
        for key, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("dtype") in ("image", "video")
    ]
    bad_images = [
        f"{row.get('episode_index')}.{key}.{field}"
        for row in rows
        for key in image_keys
        for field in ("mean", "std", "min", "max")
        if not _is_3x1x1(((row.get("stats") or {}).get(key) or {}).get(field))
    ]
    report.check(
        not bad_images,
        "episodes_stats.jsonl: image stats are (3,1,1)"
        + (f" (bad: {', '.join(bad_images[:3])})" if bad_images else ""),
    )
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            by_index[int(row.get("episode_index"))] = row
        except (TypeError, ValueError):
            report.fail("episodes_stats.jsonl: bad episode_index")
    for ep in episodes:
        idx = ep["episode_index"]
        row = by_index.get(idx)
        if not report.check(isinstance(row, dict), f"episode {idx}: stats row present"):
            continue
        block = (row.get("stats") or {}).get("observation.state") or {}
        report.check(
            block.get("count") == [ep["length"]],
            f"episode {idx}: state stats count == [length]",
        )
        names = block.get("mean")
        report.check(
            isinstance(names, list) and len(names) == dim,
            f"episode {idx}: state stats width",
        )


def _check_splits(info: dict[str, Any], episodes: list[dict[str, Any]], report: Report) -> None:
    splits = info.get("splits")
    if not report.check(isinstance(splits, dict) and bool(splits), "splits present"):
        return
    seen: list[int] = []
    bounds: dict[str, tuple[int, int]] = {}
    for name, spec in splits.items():
        try:
            start_s, stop_s = str(spec).split(":")
            start, stop = int(start_s), int(stop_s)
        except (ValueError, AttributeError):
            report.fail(f"split {name!r}: bad range {spec!r}")
            continue
        report.check(
            0 <= start < stop <= len(episodes),
            f"split {name}: {spec} inside 0..{len(episodes)}",
        )
        bounds[name] = (start, stop)
        seen.extend(range(start, stop))
    report.check(
        sorted(seen) == list(range(len(episodes))),
        "splits cover every episode exactly once",
    )
    ordered = sorted(bounds.items(), key=lambda kv: kv[1][0])
    for (name_a, (_, stop_a)), (name_b, (start_b, _)) in zip(ordered, ordered[1:]):
        report.check(
            stop_a <= start_b,
            f"splits {name_a} and {name_b} do not overlap",
        )


# --------------------------------------------------------------- with lerobot


def _check_with_lerobot(root: Path, report: Report) -> None:
    try:
        import lerobot
    except ImportError:
        report.warn("lerobot is not installed; skipping the loader check")
        return
    version = getattr(lerobot, "__version__", "unknown")
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # newer layout
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # v2.x
    except ImportError as exc:
        report.warn(f"lerobot {version}: dataset class not found ({exc})")
        return

    attempts = (
        {"repo_id": f"local/{root.name}", "root": str(root)},
        {"repo_id": f"local/{root.name}", "root": str(root), "download_videos": False},
        {"root": str(root)},
    )
    dataset = None
    signature_errors: list[str] = []
    for kwargs in attempts:
        try:
            dataset = LeRobotDataset(**kwargs)
            break
        except TypeError as exc:
            signature_errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - version-specific
            report.fail(f"lerobot {version} failed to load the dataset: {exc}")
            return
    if dataset is None:
        first = signature_errors[0] if signature_errors else "no attempt matched"
        report.warn(f"lerobot {version}: no constructor signature matched ({first})")
        return

    frames = getattr(getattr(dataset, "meta", None), "total_frames", None)
    if isinstance(frames, int):
        report.check(
            len(dataset) == frames,
            f"lerobot loads {len(dataset)} frames == meta.total_frames {frames}",
        )
    else:
        report.ok("lerobot dataset loaded")

    try:
        sample = dataset[0]
    except Exception as exc:  # noqa: BLE001 - version-specific
        report.fail(f"lerobot could not read sample 0: {exc}")
        return
    report.check("observation.state" in sample, "lerobot sample has observation.state")
    report.check("action" in sample, "lerobot sample has action")
    report.ok(f"lerobot {version} loaded the dataset")


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path, help="LeRobot dataset directory")
    ap.add_argument("--decode", action="store_true", help="decode MP4 metadata and frames")
    ap.add_argument("--with-lerobot", action="store_true", help="also load via the lerobot package")
    ap.add_argument(
        "--state-mode",
        choices=("proprio16", "touch28", "full30"),
        default=None,
        help="optionally assert the exported state width/names match this rung",
    )
    args = ap.parse_args()

    root = args.root.expanduser()
    report = Report()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 2

    print(f"checking {root}")
    info = _json_file(root / "meta" / "info.json", report)
    if not isinstance(info, dict):
        print("ERROR: no usable meta/info.json; cannot continue")
        return 1
    try:
        chunks_size = int(info["chunks_size"])
        fps = float(info["fps"])
    except (KeyError, TypeError, ValueError):
        print("ERROR: info.json needs numeric chunks_size and fps")
        return 1
    if chunks_size <= 0 or fps <= 0:
        print("ERROR: chunks_size and fps must be positive")
        return 1

    features = _check_info(info, report)
    tasks = _check_tasks(_jsonl_file(root / "meta" / "tasks.jsonl", report) or [], info, report)
    episodes = _check_episodes_meta(
        _jsonl_file(root / "meta" / "episodes.jsonl", report) or [],
        info,
        tasks,
        report,
    )
    dim = state_dim_from_info(info)
    if args.state_mode is not None:
        want = state_dim(args.state_mode)
        report.check(dim == want, f"state width {dim} == {args.state_mode} ({want})")
        names = ((features.get("observation.state") or {}).get("names")) or []
        report.check(names == STATE_NAMES[args.state_mode], f"state names match {args.state_mode}")

    index0 = 0
    for ep in episodes:
        if not all(isinstance(ep.get(k), int) for k in ("episode_index", "length")):
            continue
        index0 = _check_parquet(
            root, ep, info, tasks, expected_index0=index0, report=report
        )
    report.check(index0 == int(info.get("total_frames", -1)), "parquet rows total == total_frames")

    _check_videos(root, episodes, info, decode=bool(args.decode), report=report)
    _check_stats(root, info, episodes, report)
    _check_splits(info, episodes, report)

    if args.with_lerobot:
        _check_with_lerobot(root, report)

    print()
    print(f"{report.passed} checks passed, {len(report.warnings)} warnings, {len(report.issues)} issues")
    if report.issues:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
