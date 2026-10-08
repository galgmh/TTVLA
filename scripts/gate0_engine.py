#!/usr/bin/env python
"""Gate 0 — engine + assets smoke test (run this first).

Verifies, in order:

  1. python / numpy / h5py / torch versions and machine limits
  2. genesis imports and reports its version
  3. the assets directory contains the Panda URDF (else prints the exact
     sparse-checkout recovery commands)
  4. gs.init() works on the GPU backend (falls back to CPU with a warning)
  5. a minimal scene builds and steps
  6. every sensor class + option field our code depends on exists

Exit code 0 = all checks passed. Nothing here touches the dataset code paths,
so a failure at this gate means: fix the environment before anything else.

    python scripts/gate0_engine.py
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_common import Gate, meminfo  # noqa: E402

# Sensor classes we build, and the option fields we pass to each. Fields listed
# as forbidden must NOT exist (pydantic is extra="forbid": passing them raises
# at construction time).
SENSOR_CONTRACT: dict[str, tuple[set[str], set[str]]] = {
    "ContactProbe": ({"probe_local_pos", "probe_radius"}, {"probe_local_normal"}),
    "ContactDepthProbe": ({"probe_local_pos", "probe_radius"}, {"probe_local_normal"}),
    "KinematicTaxel": ({"probe_local_pos", "probe_radius"}, {"probe_local_normal"}),
    "Contact": (set(), set()),
    "ContactForce": (set(), set()),
    "TemperatureGrid": (
        {
            "grid_size",
            "properties_dict",
            "ambient_temperature",
            "convection_coefficient",
            "simulate_all_link_temperatures",
            "sensor_time_constant",
            "contact_depth_weight",
        },
        set(),
    ),
}


def check_environment(gate: Gate) -> None:
    gate.section("1. environment")
    gate.ok("python", f"{platform.python_version()} ({platform.system()} {platform.release()})")
    for mod in ("numpy", "h5py", "yaml", "torch"):
        try:
            m = __import__(mod)
            gate.ok(mod, getattr(m, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            gate.fail(mod, f"import failed: {exc}")
    mem = meminfo()
    if mem:
        gate.ok(
            "memory",
            f"total {mem.get('MemTotal', 0):.1f} GiB / available {mem.get('MemAvailable', 0):.1f} GiB",
        )
        if mem.get("MemAvailable", 99.0) < 4.0:
            gate.warn("memory", "less than 4 GiB available; close other WSL processes before generating")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gate.ok(
                "cuda",
                f"{props.name} | sm_{props.major}{props.minor} | {props.total_memory / 1024**3:.1f} GiB | "
                f"torch cuda {torch.version.cuda}",
            )
        else:
            gate.fail("cuda", "torch.cuda.is_available() is False")
    except Exception as exc:  # noqa: BLE001
        gate.fail("cuda", f"torch unavailable: {exc}")


def check_genesis(gate: Gate):
    gate.section("2. genesis import")
    try:
        import genesis as gs

        version = getattr(gs, "__version__", None)
        if version is None:
            try:
                from importlib.metadata import version as _v

                version = _v("genesis-world")
            except Exception:  # noqa: BLE001
                version = "?"
        gate.ok("genesis", f"version {version} | module {Path(gs.__file__).parent}")
        return gs
    except Exception as exc:  # noqa: BLE001
        gate.fail("genesis", f"import failed: {exc}")
        traceback.print_exc()
        return None


def check_assets(gate: Gate, gs) -> Path | None:
    gate.section("3. assets / robot URDF")
    try:
        assets = Path(gs.utils.get_assets_dir())
    except Exception as exc:  # noqa: BLE001
        gate.fail("assets dir", f"{exc}")
        return None
    gate.ok("assets dir", str(assets))
    for sub in ("urdf", "xml", "meshes", "textures"):
        d = assets / sub
        n = sum(1 for _ in d.rglob("*")) if d.is_dir() else -1
        if n < 0:
            gate.warn(f"assets/{sub}", "missing")
        else:
            gate.ok(f"assets/{sub}", f"{n} entries")

    urdf = assets / "urdf/panda_bullet/panda.urdf"
    if urdf.is_file():
        gate.ok("panda urdf", str(urdf))
        return urdf

    gate.fail("panda urdf", f"not found at {urdf}")
    others = sorted(str(p) for p in assets.rglob("*panda*.urdf"))[:5]
    if others:
        gate.warn("panda urdf", f"but found: {others}")
    print(
        "\n  --- recovery: restore the upstream assets (Genesis v1.4.1) ---\n"
        f"  ASSETS={assets}\n"
        "  cd /home/gmh/tactile-genesis\n"
        "  git clone --filter=blob:none --no-checkout https://github.com/Genesis-Embodied-AI/Genesis.git Genesis-assets\n"
        "  cd Genesis-assets\n"
        "  git sparse-checkout init --cone\n"
        "  git sparse-checkout set genesis/assets/urdf genesis/assets/xml genesis/assets/meshes\n"
        "  git checkout a68b97b9363aa0be9ccb0c8291b021f3afdd807d\n"
        '  cp -rn genesis/assets/urdf genesis/assets/xml genesis/assets/meshes "$ASSETS/"\n'
        "  --- then re-run this gate ---"
    )
    return None


def check_init(gate: Gate, gs) -> str | None:
    gate.section("4. gs.init")
    for backend, name in ((gs.gpu, "gpu"), (gs.cpu, "cpu")):
        try:
            gs.init(backend=backend, precision="32", seed=0, logging_level="warning")
            gate.ok("gs.init", f"backend={name}")
            if name == "cpu":
                gate.warn("backend", "running on CPU: generation will be very slow")
            return name
        except Exception as exc:  # noqa: BLE001
            gate.fail(f"gs.init[{name}]", f"{type(exc).__name__}: {exc}")
    return None


def _to_np(value):
    """Engine getters return torch tensors (CUDA on the gpu backend) -> numpy."""
    import numpy as np

    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


def check_min_scene(gate: Gate, gs) -> None:
    gate.section("5. minimal scene build + step")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 200.0, substeps=1),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.5)))
        scene.build(n_envs=1)
        z0 = float(_to_np(box.get_pos()).reshape(-1)[2])
        for _ in range(50):
            scene.step()
        z1 = float(_to_np(box.get_pos()).reshape(-1)[2])
        if z1 < z0:
            gate.ok("free fall", f"z {z0:.3f} -> {z1:.3f} m over 50 steps")
        else:
            gate.fail("free fall", f"box did not fall (z {z0:.3f} -> {z1:.3f})")
    except Exception as exc:  # noqa: BLE001
        gate.fail("minimal scene", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _options_class(gs, name: str):
    try:
        import genesis.options.sensors as gso

        cls = getattr(gso, name, None)
        if cls is not None:
            return cls
    except Exception:  # noqa: BLE001
        pass
    entity = getattr(gs.sensors, name, None)
    return getattr(entity, "Options", None)


def _fields(cls) -> set[str]:
    fields = getattr(cls, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields)
    return set(getattr(cls, "__annotations__", {}) or {})


def _entity_class(gs, name: str):
    """Options name -> runtime sensor class: ``X`` -> ``XSensor`` in
    ``genesis.engine.sensors`` (``gs.sensors.X`` is the *Options* class, not
    the object ``scene.add_sensor`` returns)."""
    import inspect

    try:
        import genesis.engine.sensors as ges
    except Exception:  # noqa: BLE001
        return None
    target = f"{name}Sensor"
    for mod_name in dir(ges):
        mod = getattr(ges, mod_name)
        if not inspect.ismodule(mod):
            continue
        cls = getattr(mod, target, None)
        if isinstance(cls, type):
            return cls
    return None


def _class_mentions(cls, attr: str) -> bool:
    """True if any method/property of ``cls`` references ``attr`` in its code.

    Instance attributes (``self._cache_size = ...``) are invisible to
    ``hasattr(cls, ...)``; scanning the code objects catches them and fails
    loudly when the engine renames an internal our GT-seeding writes through.
    """
    for member in vars(cls).values():
        for fn in (member, getattr(member, "fget", None), getattr(member, "fset", None)):
            code = getattr(fn, "__code__", None)
            if code is not None and attr in code.co_names:
                return True
    return False


# Methods our code calls on the objects scene.add_sensor() returns.
ENTITY_CONTRACT: dict[str, set[str]] = {
    "ContactProbe": {"read"},
    "ContactDepthProbe": {"read"},
    "KinematicTaxel": {"read"},
    "Contact": {"read"},
    "ContactForce": {"read"},
    "TemperatureGrid": {"read", "read_ground_truth", "link_temperatures"},
}


def check_sensor_contract(gate: Gate, gs) -> None:
    gate.section("6. sensor API contract")
    for name, (required, forbidden) in SENSOR_CONTRACT.items():
        if not hasattr(gs.sensors, name):
            gate.fail(f"gs.sensors.{name}", "missing")
            continue
        cls = _options_class(gs, name)
        if cls is None:
            gate.warn(f"gs.sensors.{name}.Options", "could not introspect fields")
            continue
        fields = _fields(cls)
        missing = required - fields
        present_forbidden = forbidden & fields
        if missing or present_forbidden:
            gate.fail(
                f"{name} options",
                f"missing={sorted(missing)} forbidden-but-present={sorted(present_forbidden)}",
            )
        else:
            gate.ok(f"{name} options", f"{len(fields)} fields")

    try:
        from genesis.options.sensors import TemperatureProperties

        names = set(TemperatureProperties._fields)
        expected = {"base_temperature", "conductivity", "density", "specific_heat", "emissivity"}
        if expected <= names:
            gate.ok("TemperatureProperties", ", ".join(TemperatureProperties._fields))
        else:
            gate.fail("TemperatureProperties", f"missing {sorted(expected - names)}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("TemperatureProperties", f"{type(exc).__name__}: {exc}")

    for name, methods in ENTITY_CONTRACT.items():
        cls = _entity_class(gs, name)
        if cls is None:
            gate.fail(f"{name}Sensor", f"entity class not found in genesis.engine.sensors ({name}Sensor)")
            continue
        missing = sorted(m for m in methods if not hasattr(cls, m))
        if missing:
            gate.fail(f"{name}Sensor reads", f"missing {missing}")
        else:
            gate.ok(f"{name}Sensor reads", ", ".join(sorted(methods)))

    # Internals our per-episode GT seeding writes through (no public setter
    # exists — see tempvla/sensors/temperature.py::seed_temperature, R1).
    try:
        import dataclasses

        from genesis.engine.sensors.base_sensor import Sensor
        from genesis.engine.sensors.sensor_manager import SensorManager
        from genesis.engine.sensors.temperature import TemperatureGridSensorMetadata

        internals = {
            "SensorManager": (SensorManager, ("_ground_truth_intermediate_cache", "_cache_slices_by_type")),
            "Sensor": (Sensor, ("_cache_size", "_cache_offset", "_shared_metadata", "_idx")),
            "Scene": (gs.Scene, ("_sanitize_envs_idx",)),
        }
        missing = []
        for label, (cls, attrs) in internals.items():
            for attr in attrs:
                if not (_class_mentions(cls, attr) or hasattr(cls, attr)):
                    missing.append(f"{label}.{attr}")
        # Dataclass fields (default_factory fields have no class attribute).
        meta_fields = {f.name for f in dataclasses.fields(TemperatureGridSensorMetadata)}
        for attr in ("link_temps", "links_idx", "cache_sizes"):
            if attr not in meta_fields:
                missing.append(f"TemperatureGridSensorMetadata.{attr}")
        if missing:
            gate.fail("temperature seeding internals", f"missing {missing}")
        else:
            gate.ok("temperature seeding internals", "GT cache + link_temps + cache slices present")
    except Exception as exc:  # noqa: BLE001
        gate.fail("temperature seeding internals", f"{type(exc).__name__}: {exc}")

    try:
        from tempvla.config import TaskConfig
        from tempvla.scene import resolve_urdf_path

        gate.ok("tempvla config", f"urdf resolves to {resolve_urdf_path(TaskConfig())}")
    except Exception as exc:  # noqa: BLE001
        gate.fail("tempvla import", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-scene", action="store_true", help="skip the build/step check")
    args = parser.parse_args()

    gate = Gate("Gate 0 — engine + assets")
    check_environment(gate)
    gs = check_genesis(gate)
    if gs is not None:
        check_assets(gate, gs)
        if check_init(gate, gs) is not None:
            if not args.skip_scene:
                check_min_scene(gate, gs)
            check_sensor_contract(gate, gs)
    return gate.summary("Gate 0 passed — proceed to: python scripts/gate1_sensors.py")


if __name__ == "__main__":
    raise SystemExit(main())
