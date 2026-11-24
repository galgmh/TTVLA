"""Run and record one already-constructed simulator episode."""

from __future__ import annotations

import numpy as np

from sim.dataset import metadata_for_spec
from sim.episode_recorder import EpisodeRecorder
from sim.control.probe_controller import MultiProbeController, ProbeController
from sim.obs import get_policy_observation, get_privileged_observation, get_robot_observation
from sim.task_success import evaluate_task_success


def run_episode(bundle, spec, *, controller=None) -> tuple[dict, dict, dict]:
    probe = controller or (
        MultiProbeController(bundle, spec) if spec.contact_order else ProbeController(bundle, spec)
    )
    recorder = EpisodeRecorder(target_region=int(spec.target_region), link_count=len(bundle.regions))
    recorder.set_camera_calibration(bundle.camera_calibration())
    sim_dt = float(bundle.cfg["controller"]["sim_dt"])

    def on_sensor_sample(sim_step: int) -> None:
        recorder.add_sensor(
            sim_time=float(sim_step) * sim_dt,
            tactile=get_policy_observation(bundle),
            privileged=get_privileged_observation(bundle),
        )

    def on_control_sample(sim_step: int) -> None:
        command = probe.last_command
        if command is None:
            raise RuntimeError("控制器尚未下发任何指令")
        recorder.add_control(
            sim_time=float(sim_step) * sim_dt,
            sensor_index=recorder.sensor_frames - 1,
            command=command,
            robot=get_robot_observation(bundle),
            camera=bundle.camera_observation(),
        )

    initial_cup_z = float(bundle.cup_pose()[0][2])
    outcome = probe.run(on_sample=on_sensor_sample, on_control=on_control_sample)
    recorder.finalize()
    cup_pos, cup_quat = bundle.cup_pose()
    import genesis as gs

    metadata = metadata_for_spec(
        spec,
        genesis_version=str(getattr(gs, "__version__", "unknown")),
        cup_pose=np.concatenate([cup_pos, cup_quat]),
        cup_asset_version=str(bundle.cfg["object"]["asset_version"]),
        outcome=outcome,
    )
    metadata.update(recorder.summary())
    metadata["task_id"] = str(bundle.cfg["task_id"])
    metadata["task_success"] = evaluate_task_success(
        bundle.cfg,
        recorder.buffers,
        initial_cup_z=initial_cup_z,
        final_cup_z=float(cup_pos[2]),
        region_order=spec.region_order,
        initial_link_temps_c=spec.link_temps_c,
    )
    metadata["config_paths"] = dict(bundle.cfg["config_paths"])
    return recorder.buffers, metadata, recorder.static


__all__ = ["run_episode"]
