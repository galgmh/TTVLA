#!/usr/bin/env python
"""Build training-ready sidecar manifests for a dataset directory.

Canonical data stays robomimic HDF5 (format ``tempvla-1.x``); what VLA
training / validation / evaluation need on top are three small sidecar files
(design + rationale: docs/architecture.md §9-R30 in git HEAD; dataset card:
datasets/README.md):

    index.jsonl    one line per episode: file, demo, seed, split, plus every
                   recorded per-episode attr (JSON-safe scalars/short arrays)
    splits.json    train/val/test assignment. The split is a pure function of
                   the episode seed -- sha1("temperaturevla-split-v1:<seed>")
                   mod 10000 -> 80/10/10 by default -- so it is stable across
                   runs, machines, data dirs and incremental generation
    stats.json     per-key normalization statistics computed on the TRAIN
                   split only (computing them on val/test leaks). Small keys
                   get mean/std/min/max/q01/q99 per element; large float keys
                   get streaming abs-max/nonzero stats; non-float keys (bool
                   contact, uint8 images) are inventoried but not reduced

Pure h5py/numpy/stdlib -- no genesis import, so a dataloader-side machine can
run it too (same rule as sim/tools/).

Usage (tempvla env):
    python scripts/build_manifest.py datasets/hdf5/pick_cold_can
    python scripts/build_manifest.py <dir> --val-frac 0.1 --test-frac 0.1   # split fractions
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

MANIFEST_VERSION = 1
SPLIT_SALT = "temperaturevla-split-v1"  # changing this re-partitions every dataset
QUANTILE_MAX_ELEMS = 512   # per-sample elements below which full quantiles are kept
ARRAY_ATTR_MAX = 16        # per-episode array attrs longer than this are skipped


def split_of(seed: int, *, val_frac: float, test_frac: float) -> str:
    """Deterministic split assignment -- a pure function of the episode seed."""
    digest = hashlib.sha1(f"{SPLIT_SALT}:{int(seed)}".encode("ascii")).hexdigest()
    u = int(digest[:8], 16) % 10000 / 10000.0
    if u < 1.0 - val_frac - test_frac:
        return "train"
    if u < 1.0 - test_frac:
        return "val"
    return "test"


def _jsonable(v):
    """Convert an HDF5 attr to a JSON-safe value, or None to skip it."""
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "OUS" or v.size > ARRAY_ATTR_MAX:
            return None
        return v.tolist()
    if isinstance(v, (bool, int, float)):
        return v
    return None


def _iter_leaves(group: h5py.Group, prefix: str = ""):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            yield from _iter_leaves(item, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", item


def _demos(f: h5py.File) -> list[str]:
    return sorted(f["data"].keys(), key=lambda s: int(s.rsplit("_", 1)[-1]))


def build(data_dir: Path, out_dir: Path, val_frac: float, test_frac: float) -> int:
    files = sorted(p for p in data_dir.glob("*.hdf5"))
    if not files:
        print(f"no *.hdf5 files under {data_dir}")
        return 1

    warnings: list[str] = []
    rows: list[dict] = []
    file_meta: list[dict] = []
    n_failed_files = 0

    # ---- pass 1: index + splits -------------------------------------------
    for path in files:
        with h5py.File(path, "r") as f:
            meta = {"name": path.name, "kind": "failed" if "_failed" in path.stem else "ok"}
            for key in ("format_version", "generator", "task_id", "problem_info"):
                value = f.attrs.get(key)
                if value is not None:
                    meta[key] = _jsonable(value)
            demos = _demos(f)
            meta["n_demos"] = len(demos)
            file_meta.append(meta)
            n_failed_files += meta["kind"] == "failed"
            for idx, demo in enumerate(demos):
                attrs = {k: _jsonable(v) for k, v in f["data"][demo].attrs.items()}
                attrs = {k: v for k, v in attrs.items() if v is not None}
                seed = f["data"][demo].attrs.get("episode_seed")
                if seed is None:
                    warnings.append(f"{path.name}/{demo}: no episode_seed, using demo index")
                    seed = idx
                attrs.pop("episode_seed", None)
                row = {
                    "kind": "episode",
                    "file": path.name,
                    "file_kind": meta["kind"],
                    "demo": demo,
                    "seed": int(seed),
                    "split": split_of(int(seed), val_frac=val_frac, test_frac=test_frac),
                }
                row.update(attrs)
                rows.append(row)

    ok_rows = [r for r in rows if r["file_kind"] == "ok"]
    train_rows = [r for r in ok_rows if r["split"] == "train" and r.get("success")]
    seeds = {"train": [], "val": [], "test": []}
    for seed in sorted({r["seed"] for r in ok_rows}):
        seeds[split_of(seed, val_frac=val_frac, test_frac=test_frac)].append(seed)

    # ---- pass 2: stats on the train split only ----------------------------
    inventory: dict[str, dict] = {}
    acc: dict[str, dict] = {}

    def _touch(key: str, ds: h5py.Dataset, demo_ref: str) -> None:
        per_sample = tuple(ds.shape[1:])
        inv = inventory.setdefault(key, {"shape": list(per_sample), "dtype": str(ds.dtype), "n_episodes": 0})
        inv["n_episodes"] += 1
        if tuple(inv["shape"]) != per_sample:
            warnings.append(f"{key}: shape drift {inv['shape']} vs {per_sample} ({demo_ref})")
            # permissive: the first-seen shape stays in the inventory
        if ds.dtype.kind != "f":
            return
        a = acc.get(key)
        if a is None:
            a = acc[key] = {"per_sample": per_sample, "n_ep": 0, "n_samp": 0, "buf": [],
                            "sum": 0.0, "sumsq": 0.0, "min": None, "max": None,
                            "absmax": 0.0, "nonzero": 0, "frames": 0}
        a["n_ep"] += 1
        a["n_samp"] += int(ds.shape[0])
        if np.prod(per_sample) <= QUANTILE_MAX_ELEMS:
            a["buf"].append(np.asarray(ds[:], dtype=np.float64))
        else:
            arr = np.asarray(ds[:], dtype=np.float64)
            a["sum"] = a["sum"] + arr.sum(axis=0)
            a["sumsq"] = a["sumsq"] + np.square(arr).sum(axis=0)
            a["min"] = arr.min(axis=0) if a["min"] is None else np.minimum(a["min"], arr.min(axis=0))
            a["max"] = arr.max(axis=0) if a["max"] is None else np.maximum(a["max"], arr.max(axis=0))
            a["absmax"] = max(a["absmax"], float(np.abs(arr).max()))
            a["nonzero"] += int(np.count_nonzero(arr))
            a["frames"] += int(arr.shape[0])

    for path in files:
        if "_failed" in path.stem:
            continue
        want = {(r["file"], r["demo"]) for r in train_rows if r["file"] == path.name}
        if not want:
            continue
        with h5py.File(path, "r") as f:
            for demo in _demos(f):
                if (path.name, demo) not in want:
                    continue
                d = f["data"][demo]
                _touch("actions", d["actions"], f"{path.name}/{demo}")
                for key, ds in _iter_leaves(d["obs"]):
                    _touch(f"obs/{key}", ds, f"{path.name}/{demo}")

    keys_stats: dict[str, dict] = {}
    for key in sorted(acc):
        a = acc[key]
        entry = {"shape": list(a["per_sample"]), "n_episodes": a["n_ep"], "n_samples": a["n_samp"]}
        if a["buf"]:
            arr = np.concatenate(a["buf"], axis=0)
            entry.update({
                "mean": np.round(arr.mean(axis=0), 6).tolist(),
                "std": np.round(arr.std(axis=0), 6).tolist(),
                "min": np.round(arr.min(axis=0), 6).tolist(),
                "max": np.round(arr.max(axis=0), 6).tolist(),
                "q01": np.round(np.quantile(arr, 0.01, axis=0), 6).tolist(),
                "q99": np.round(np.quantile(arr, 0.99, axis=0), 6).tolist(),
                "absmax": float(np.abs(arr).max()),
                "nonzero_frac": float(np.count_nonzero(arr) / arr.size),
            })
        else:
            frames = max(a["frames"], 1)
            mean = a["sum"] / frames
            var = np.maximum(a["sumsq"] / frames - np.square(mean), 0.0)
            entry.update({
                "mean": np.round(mean, 6).tolist(),
                "std": np.round(np.sqrt(var), 6).tolist(),
                "min": np.round(a["min"], 6).tolist(),
                "max": np.round(a["max"], 6).tolist(),
                "absmax": float(a["absmax"]),
                "nonzero_frac": float(a["nonzero"] / (frames * int(np.prod(a["per_sample"])))),
                "note": "streaming stats (per-sample size > %d): no quantiles" % QUANTILE_MAX_ELEMS,
            })
        keys_stats[key] = entry

    # ---- write -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    with index_path.open("w") as fh:
        header = {"kind": "manifest", "manifest_version": MANIFEST_VERSION,
                  "data_dir": str(data_dir), "files": file_meta}
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    splits = {
        "manifest_version": MANIFEST_VERSION,
        "rule": {"salt": SPLIT_SALT, "hash": "sha1", "mod": 10000,
                 "val_frac": val_frac, "test_frac": test_frac},
        "counts": {k: len(v) for k, v in seeds.items()},
        "episodes": {k: sum(r["split"] == k for r in ok_rows) for k in seeds},
        "seeds": seeds,
        "note": "assignment is a pure function of episode_seed and excludes *_failed.hdf5; "
                "a seed's split is identical in every data dir built with the same rule",
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n")

    stats = {
        "manifest_version": MANIFEST_VERSION,
        "computed_on": "train split, non-failed files, success=True episodes only",
        "n_episodes": len(train_rows),
        "keys": keys_stats,
        "inventory": {k: {"shape": v["shape"], "dtype": v["dtype"], "n_episodes": v["n_episodes"]}
                      for k, v in sorted(inventory.items())},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    # ---- report ------------------------------------------------------------
    print(f"build_manifest: {data_dir}")
    print(f"  files: {len(files)} ({n_failed_files} failed) | episodes: {len(rows)} "
          f"({len(ok_rows)} ok / {len(rows) - len(ok_rows)} failed)")
    print(f"  splits (seeds): train {splits['counts']['train']}, val {splits['counts']['val']}, "
          f"test {splits['counts']['test']}  ->  episodes: {splits['episodes']}")
    print(f"  stats: {len(keys_stats)} float keys on {len(train_rows)} train episodes | "
          f"inventory: {len(inventory)} channel keys")
    print(f"  wrote: {index_path.name}, splits.json, stats.json -> {out_dir}")
    for w in warnings[:10]:
        print(f"  [warn] {w}")
    if len(warnings) > 10:
        print(f"  [warn] ... {len(warnings) - 10} more")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory containing the HDF5 files")
    ap.add_argument("--out", type=Path, default=None,
                    help="sidecar output directory (default: same as data_dir)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    return build(args.data_dir, args.out or args.data_dir, args.val_frac, args.test_frac)


if __name__ == "__main__":
    raise SystemExit(main())
