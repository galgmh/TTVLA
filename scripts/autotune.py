#!/usr/bin/env python
"""Automatic expert-parameter tuning (random search + local refinement).

Runs short episode batches with candidate expert parameters and scores them on
success rate (primary) plus signal richness (contact events, fingertip heating)
and IK tracking quality. The best parameters are written back into a copy of
the task YAML, which generation can then use directly:

    python scripts/autotune.py --trials 24 --episodes 5
    python scripts/generate.py --config configs/pick_hot_cylinder.tuned.yaml --episodes 50

Design notes
------------
* Evaluation episodes use the **same seeds as generate.py** (``cfg.seed + i``)
  and the same scene seed (``cfg.seed + 1000 * (material_index + 1)``), so the
  measured success rate is the expected success rate of the real generation
  run — not an artifact of easier episodes.
* Controller scales (``pos_scale_m`` / ``rot_scale_rad``) are deliberately NOT
  tuned: they define the dataset action space and must stay fixed.
* No HDF5 is written during tuning (``writer=None``), so it is much faster than
  generation and leaves no partial files behind.
* Every trial is appended to ``<out>.trials.jsonl``; rerun with ``--resume`` to
  skip finished random-phase trials after an interruption.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import load_config  # noqa: E402

# Search space: name -> (low, high). Keys must be ExpertConfig fields.
PARAM_SPACE: dict[str, tuple[float, float]] = {
    "approach_clearance_m": (0.04, 0.15),
    "grasp_z_offset_m": (-0.010, 0.015),
    "pos_tol_m": (0.005, 0.020),
    "rot_tol_rad": (0.10, 0.35),
    "descend_tol_m": (0.008, 0.025),
    "contact_depth_m": (1e-6, 1e-4),
    "squeeze_steps": (3, 25),
    "close_timeout_steps": (20, 80),
    "lift_height_m": (0.06, 0.15),
    "lift_tol_m": (0.008, 0.030),
    "grasp_xy_jitter_m": (0.0, 0.008),
    "approach_yaw_jitter_rad": (0.0, 0.30),
    "close_extra": (0.8, 1.0),
}
INT_PARAMS = {"squeeze_steps", "close_timeout_steps"}
LOG_PARAMS = {"contact_depth_m"}  # sampled/perturbed in log space


# --------------------------------------------------------------------- search


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (lo, hi) in PARAM_SPACE.items():
        if name in LOG_PARAMS:
            value = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            value = float(rng.uniform(lo, hi))
        out[name] = int(round(value)) if name in INT_PARAMS else value
    return out


def perturb_params(rng: np.random.Generator, params: dict[str, float], sigma_frac: float) -> dict[str, float]:
    """Move 1-3 random coordinates of ``params`` by sigma_frac * range."""
    out = dict(params)
    k = int(rng.integers(1, 4))
    for name in rng.choice(list(PARAM_SPACE), size=k, replace=False):
        lo, hi = PARAM_SPACE[name]
        if name in LOG_PARAMS:
            lo_l, hi_l = np.log(lo), np.log(hi)
            value = float(np.exp(np.clip(np.log(out[name]) + rng.normal(0.0, sigma_frac * (hi_l - lo_l)), lo_l, hi_l)))
        else:
            value = float(np.clip(out[name] + rng.normal(0.0, sigma_frac * (hi - lo)), lo, hi))
        out[name] = int(round(value)) if name in INT_PARAMS else value
    return out


def score_of(agg: dict) -> float:
    """Success rate dominates; richness/tracking are small tie-breakers."""
    if not agg.get("episodes"):
        return -1.0
    return (
        agg["success_rate"]
        + 0.05 * min(agg["events"], 4.0) / 4.0
        + 0.05 * min(agg["heating_c"], 3.0) / 3.0
        - 0.05 * min(agg["ik_err_mm"], 20.0) / 20.0
    )


# ----------------------------------------------------------------- evaluation


def evaluate(cfg, runner_cls, bundle, params: dict, seeds: list[int], *, max_steps: int | None) -> dict:
    from tempvla.expert import PickExpert

    expert_cfg = dataclasses.replace(cfg.expert, **params)
    runner = runner_cls(bundle)
    runner.expert = PickExpert(
        expert_cfg,
        pos_scale_m=cfg.controller.pos_scale_m,
        rot_scale_rad=cfg.controller.rot_scale_rad,
    )

    from tempvla.recording import summarize_episode

    rows, errors = [], 0
    for seed in seeds:
        try:
            res = runner.run_episode(episode_seed=seed, max_steps=max_steps)
        except Exception as exc:  # noqa: BLE001 - a bad candidate must not kill the search
            errors += 1
            print(f"      seed {seed}: ERROR {type(exc).__name__}: {exc}")
            continue
        rows.append(summarize_episode(res))
        del res

    n = len(rows)
    if not n:
        return {"episodes": 0, "errors": errors, "success_rate": 0.0, "events": 0.0,
                "windows": 0.0, "heating_c": 0.0, "ik_err_mm": 0.0, "steps": 0.0}
    return {
        "episodes": n,
        "errors": errors,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "events": float(np.mean([r["events"] for r in rows])),
        "windows": float(np.mean([r["windows"] for r in rows])),
        "heating_c": float(np.mean([r["heating_c"] for r in rows])),
        "ik_err_mm": float(np.mean([r["ik_err_max"] for r in rows]) * 1000.0),
        "steps": float(np.mean([r["steps"] for r in rows])),
    }


# -------------------------------------------------------------------- logging


def load_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_log(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def fmt_params(params: dict) -> str:
    keys = {
        "grasp_z_offset_m": "z",
        "approach_clearance_m": "clr",
        "squeeze_steps": "sqz",
        "close_extra": "grip",
        "grasp_xy_jitter_m": "xy",
        "contact_depth_m": "depth",
    }
    return " ".join(f"{short}={params[k]:.4g}" for k, short in keys.items())


# ----------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="base task YAML")
    parser.add_argument("--material", default=None, help="material to tune on (default: first in config)")
    parser.add_argument("--trials", type=int, default=24, help="total trials (random + local)")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per trial")
    parser.add_argument("--seed", type=int, default=None, help="override cfg.seed (drives trial + episode seeds)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None, help="tuned YAML path (default: <config stem>.tuned.yaml)")
    parser.add_argument("--budget-min", type=float, default=None, help="stop after this many minutes")
    parser.add_argument("--resume", action="store_true", help="skip finished random-phase trials from the log")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    material = args.material or cfg.materials[0]
    if material not in cfg.material_properties:
        print(f"ERROR: material '{material}' not in material_properties {sorted(cfg.material_properties)}")
        return 2

    out_path = Path(args.out) if args.out else Path(args.config or "configs/pick_hot_cylinder.yaml")
    if args.out is None:
        out_path = out_path.with_suffix("").with_suffix(".tuned.yaml")
    log_path = out_path.with_suffix(".trials.jsonl")

    # Same scene seed as generate.py for this material -> same object instance.
    material_index = list(cfg.materials).index(material)
    scene_seed = cfg.seed + 1000 * (material_index + 1)
    seeds = [cfg.seed + i for i in range(args.episodes)]

    print("TemperatureVLA expert autotune")
    print(f"  config      : {args.config or '(defaults)'}")
    print(f"  material    : {material}  (scene seed {scene_seed})")
    print(f"  budget      : {args.trials} trials x {args.episodes} episodes, seeds {seeds}")
    print(f"  success     : lift > {cfg.lift_success_m * 100:.0f} cm, kept {cfg.stable_steps} steps")
    print(f"  output      : {out_path}")
    print(f"  trial log   : {log_path}")

    from tempvla.recording import EpisodeRunner
    from tempvla.scene import build_scene

    with_timer = time.perf_counter()
    bundle = build_scene(cfg, material=material, seed=scene_seed, show_viewer=args.viewer)
    print(f"  scene built in {time.perf_counter() - with_timer:.1f}s: r={bundle.object_spec.radius_m:.4f} m "
          f"h={bundle.object_spec.height_m:.4f} m m={bundle.object_spec.mass_kg:.4f} kg")

    records = load_log(log_path) if args.resume else []
    if records:
        print(f"  resume      : {len(records)} finished trials loaded from the log")
    best = max(records, key=lambda r: r["score"]) if records else None
    done_random = {r["trial"] for r in records if r["phase"] == "random"}

    n_random = max(1, int(round(args.trials * 0.4)))
    n_local = max(0, args.trials - n_random)
    t0 = time.perf_counter()
    trial_count = len(records)

    def run_trial(index: int, phase: str, params: dict) -> bool:
        """Evaluate one candidate, log it, update ``best``. False = stop."""
        nonlocal best, trial_count
        if args.budget_min is not None and (time.perf_counter() - t0) / 60.0 > args.budget_min:
            print(f"  budget of {args.budget_min:g} min reached — stopping")
            return False
        print(f"  [{phase:<6} {index:>3}] {fmt_params(params)}")
        with_ep = time.perf_counter()
        agg = evaluate(cfg, EpisodeRunner, bundle, params, seeds, max_steps=args.max_steps)
        s = score_of(agg)
        record = {"trial": index, "phase": phase, "params": params, "agg": agg, "score": s}
        append_log(log_path, record)
        trial_count += 1
        print(
            f"          score {s:+.3f} | success {agg['success_rate'] * 100:>3.0f}% "
            f"({agg['episodes']}/{args.episodes} runs) "
            f"| events {agg['events']:.1f} | heat {agg['heating_c']:+.2f}C "
            f"| ik {agg['ik_err_mm']:.1f}mm | {time.perf_counter() - with_ep:.1f}s"
        )
        if best is None or s > best["score"]:
            best = record
            print(f"          ** new best (score {s:+.3f}) **")
        return True

    print(f"\n=== random search ({n_random} trials) ===")
    for i in range(n_random):
        if i in done_random:
            continue
        params = sample_params(np.random.default_rng(cfg.seed * 7919 + i))
        if not run_trial(i, "random", params):
            break

    if best is not None and n_local:
        print(f"\n=== local refinement ({n_local} trials around the best) ===")
        for j in range(n_local):
            rng = np.random.default_rng(cfg.seed * 104729 + n_random + j)
            sigma = max(0.05, 0.30 * 0.85 ** (j // 3))
            params = perturb_params(rng, best["params"], sigma)
            if not run_trial(n_random + j, "local", params):
                break

    # ------------------------------------------------------------------- report
    print("\n=== summary ===")
    if best is None:
        print("no trials completed — nothing written")
        return 1
    print(f"trials: {trial_count} | best score {best['score']:+.3f} | "
          f"success {best['agg']['success_rate'] * 100:.0f}% | "
          f"events {best['agg']['events']:.1f} | heat {best['agg']['heating_c']:+.2f}C | "
          f"ik {best['agg']['ik_err_mm']:.1f}mm")
    print("best params:")
    for k, v in best["params"].items():
        print(f"  {k}: {v:g}")

    top = sorted(records + [best], key=lambda r: r["score"], reverse=True)[:5]
    seen, dedup = set(), []
    for r in top:
        key = json.dumps(r["params"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            dedup.append(r)

    cfg.expert = dataclasses.replace(cfg.expert, **best["params"])
    from tempvla.config import save_task_config

    save_task_config(
        cfg,
        out_path,
        extra={
            "tuning": {
                "source_config": str(args.config or ""),
                "material": material,
                "scene_seed": scene_seed,
                "episode_seeds": seeds,
                "trials": trial_count,
                "episodes_per_trial": args.episodes,
                "best_score": round(best["score"], 6),
                "best_agg": {k: round(v, 6) if isinstance(v, float) else v for k, v in best["agg"].items()},
                "top_trials": [
                    {"score": round(r["score"], 6), "agg": r["agg"], "params": r["params"]} for r in dedup
                ],
                "wall_min": round((time.perf_counter() - t0) / 60.0, 2),
            }
        },
    )
    print(f"\nwrote {out_path} (expert: section = tuned params; tuning: = provenance)")
    print(f"next: python scripts/generate.py --config {out_path} --episodes 50 --out data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
