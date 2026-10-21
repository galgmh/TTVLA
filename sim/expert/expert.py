"""Scripted pick experts: normalized 7-dim actions, policy-identical interface.

Phases of :class:`PickExpert`:

    APPROACH -> DESCEND -> CLOSE -> LIFT -> HOLD -> DONE
                  ^ tactile contact + squeeze before lifting

and of :class:`PickByTemperatureExpert`, which inserts the temperature judgement
between closing and lifting (docs §9-R38):

    APPROACH -> DESCEND -> CLOSE -> JUDGE -+-> LIFT -> HOLD -> DONE
                                           |
                                           +-> RELEASE -> APPROACH(next body) -> ...

and of :class:`PutInTrayExpert`, which keeps that whole machine and replaces the
tail with the carry to the table's tray (docs §9-R39). Every candidate is
judged: a match is carried, and the machine then returns for the candidates
that are still unjudged; a rejection is released and moves on. The episode
ends (DONE) only after the last candidate.

    ... -> JUDGE -+-> CARRY -> LOWER -> DROP -> RETREAT -+-> APPROACH(next) -> ...
                  |                                     +-> DONE
                  +-> RELEASE -+-> APPROACH(next body) -> ...
                               +-> DONE

All of them consume only the policy-rate observation dict (never the engine), so
they exercise exactly the interface a learned policy would see. Per-episode
jitter (grasp xy offset, approach yaw, closure amount) is sampled on reset to
broaden the demonstration distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from sim.expert.actions import normalized_action_to
from sim.expert.pipeline import PIPELINE, PipelineConfig, target_kind
from sim.rotations import quat_error_rotvec, quat_mul, rotvec_to_quat, yaw_of_quat, yaw_to_quat
from sim.scene import SceneBundle

# 180 deg about world x: hand z-axis points down, fingers close horizontally.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# Tray slots for successive carried bodies, metres from the cavity centre. The
# first placement keeps the centre; later matches step aside along x, which is
# the axis the fingers do *not* open along (the joints translate along the hand's
# y, so the open pads straddle the body in world y and the grasp comes down with
# its pads out of the first body's way). The offset is set by the bodies, not the
# pads: two 25 mm-radius bodies need >= 50 mm between centres, and the cavity's
# inner half-width (85 mm) leaves a body centre at most 60 mm from the centre, so
# 55 mm is the usable point -- a 45 mm slot overlapped the two bodies by 5 mm;
# in the 2026-10-21 smoke one such second drop hung in LOWER (another settled
# only after the contact nudge). Unreachable in the current config anyway (only
# body 0 answers [null, 15]); revisit this before two candidates can both match.
PLACE_SLOTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.055, 0.0),
    (-0.055, 0.0),
)


class Phase(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    HOLD = "hold"
    DONE = "done"
    # PickByTemperatureExpert only (docs §9-R38).
    JUDGE = "judge"
    RELEASE = "release"
    # PutInTrayExpert only (docs §9-R39): the carry-and-place tail that replaces
    # LIFT -> HOLD once the judgement is positive.
    CARRY = "carry"
    LOWER = "lower"
    DROP = "drop"
    RETREAT = "retreat"


@dataclass
class ExpertState:
    phase: str = Phase.APPROACH.value
    steps_in_phase: int = 0
    contact_seen: bool = False
    squeeze_count: int = 0
    grasp_confirmed: bool = False
    jitter: dict[str, Any] = field(default_factory=dict)
    # APPROACH sub-step when ``pipeline.transit_clearance_m > 0`` (docs §9-R43):
    # 0 = rise straight up, 1 = cross at transit height, 2 = descend onto the
    # approach point. Unused (and harmless) with the transit disabled.
    approach_stage: int = 0


@dataclass
class JudgeState(ExpertState):
    """ExpertState plus the temperature judge's per-grasp bookkeeping.

    ``cand`` is the expert's own candidate index and deliberately does *not*
    touch ``bundle.target_index`` (that one is the episode's ground-truth target
    and drives both success and the recorded attrs). ``onset_s`` is per finger:
    the two pads do not touch at the same control step.
    """

    cand: int = 0
    onset_s: list[float | None] = field(default_factory=list)
    # One record per touch: the measured fingertip evidence and the decision
    # that ended the hold (interval reached / trend away / unmeasured).
    temperature_log: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "idle"  # idle | collecting | done
    retry: int = 0  # re-grasps already spent on ``cand`` (pipeline.retry_on_none)
    rejudge: bool = False  # this RELEASE re-approaches the same body, not the next


@dataclass
class PlaceState(JudgeState):
    """JudgeState plus the carry-and-place tail's bookkeeping (docs §9-R39).

    ``stage`` is the sub-step inside CARRY (raise clear of the table first, then
    translate to the tray); ``place`` records what the current placement aimed
    at, which the episode's attrs carry so a recording says where the body was
    meant to go and which one it was. ``places`` is every placement of the
    episode in order -- every candidate that matched is carried, so there can be
    more than one.
    """

    stage: int = 0
    place: dict[str, Any] = field(default_factory=dict)
    places: list[dict[str, Any]] = field(default_factory=list)


class PickExpert:
    """Deterministic pick state machine with a small randomized jitter."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        pipeline: PipelineConfig | None = None,
        pos_scale_m: float = 0.05,
        rot_scale_rad: float = 0.5,
    ):
        self.cfg = cfg
        self.pipe = pipeline or PIPELINE
        self.pos_scale_m = float(pos_scale_m)
        self.rot_scale_rad = float(rot_scale_rad)
        self.state = ExpertState()
        self.grasp_pos = np.zeros(3)
        self.grasp_quat = TOP_DOWN_QUAT.copy()
        self.target_pos = np.zeros(3)
        self._bundle: SceneBundle | None = None

    # ------------------------------------------------------------------- setup

    def reset(self, bundle: SceneBundle, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        self._bundle = bundle
        jitter = self._set_goal(bundle, bundle.target_index, rng)
        self.state = ExpertState(jitter=jitter)

    # -------------------------------------------------------------- goal poses

    def _set_goal(self, bundle: SceneBundle, index: int, rng: np.random.Generator) -> dict[str, Any]:
        """Aim the hand at object ``index`` and return the jitter it drew.

        Split out of :meth:`reset` so :class:`PickByTemperatureExpert` can re-aim
        at another body mid-episode. For the target index the pose read is
        ``bundle.object_pose()``, i.e. exactly what ``reset`` always used, so the
        existing tasks' episodes are unchanged.
        """
        pipe = self.pipe
        xy_jitter = rng.uniform(-pipe.grasp_xy_jitter_m, pipe.grasp_xy_jitter_m, size=2)
        yaw_jitter = float(rng.uniform(-pipe.approach_yaw_jitter_rad, pipe.approach_yaw_jitter_rad))

        obj_pos, obj_quat = bundle.object_pose_of(int(index))
        self.grasp_pos = np.array(
            [obj_pos[0] + xy_jitter[0], obj_pos[1] + xy_jitter[1], obj_pos[2] + pipe.grasp_z_offset_m]
        )
        approach_yaw = yaw_jitter
        if str(bundle.object_specs[int(index)].shape).lower() in ("box", "cube"):
            # A box — or a cube, which is the same thing with equal sides — is
            # only graspable face-on: the fingers must close along a face
            # normal, or they land on an edge/corner. The jitter picks which face
            # (nearest 90 deg multiple), then the object's own body yaw is added
            # so the fingers end up exactly parallel to that face. Quantizing
            # (object_yaw + jitter) instead would align to the *world* axes and
            # leave a residual equal to the object yaw (up to +-17 deg here),
            # which wedges the pads on an edge: uneven marker field, weak press.
            object_yaw = yaw_of_quat(obj_quat)
            approach_yaw = round(yaw_jitter / (np.pi / 2.0)) * (np.pi / 2.0) + object_yaw
        self.grasp_quat = quat_mul(yaw_to_quat(approach_yaw), TOP_DOWN_QUAT)
        self.target_pos = self.grasp_pos + np.array([0.0, 0.0, pipe.approach_clearance_m])
        return {"xy": xy_jitter.tolist(), "yaw": approach_yaw}

    # -------------------------------------------------------------------- step

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        pipe = self.pipe
        st = self.state
        st.steps_in_phase += 1

        cur_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        cur_quat = rotvec_to_quat(np.asarray(obs["ee_ori"], dtype=np.float64))

        depth = np.asarray(obs["tactile"].get("depth", []), dtype=np.float64)
        in_contact = bool(np.any(depth > pipe.contact_depth_m)) or bool(np.any(obs["contact"]["link_contact"]))

        grip = -1.0
        if st.phase == Phase.APPROACH.value:
            grip = self._act_approach(st, cur_pos, cur_quat)

        elif st.phase == Phase.DESCEND.value:
            self.target_pos = self.grasp_pos
            # Measured against the target just assigned, not against the previous
            # step's setpoint: that stale error is below `descend_tol_m` the
            # moment APPROACH finishes (pos_tol_m < descend_tol_m), which would
            # skip the descent entirely and close the pads while still 8 cm high.
            if float(np.linalg.norm(self.grasp_pos - cur_pos)) < pipe.descend_tol_m:
                self._advance(Phase.CLOSE)

        elif st.phase == Phase.CLOSE.value:
            self.target_pos = self.grasp_pos
            grip = pipe.close_extra
            if in_contact and not st.contact_seen:
                st.contact_seen = True
                st.squeeze_count = 0
            if st.contact_seen:
                st.squeeze_count += 1
                if st.squeeze_count >= pipe.squeeze_steps:
                    st.grasp_confirmed = True
                    self._advance(self._on_grasped())
            elif st.steps_in_phase >= pipe.close_timeout_steps:
                self._advance(self._on_close_timeout())

        elif st.phase in (Phase.LIFT.value, Phase.HOLD.value):
            lift_pos = self.grasp_pos + np.array([0.0, 0.0, pipe.lift_height_m])
            self.target_pos = lift_pos
            grip = pipe.close_extra
            # Same fresh-target rule as DESCEND: the previous setpoint is ~0 away
            # on the step LIFT starts, so the old check left HOLD to do the
            # lifting. The gate is now what its name says.
            if st.phase == Phase.LIFT.value and float(
                np.linalg.norm(lift_pos - cur_pos)
            ) < pipe.lift_tol_m:
                self._advance(Phase.HOLD)

        elif st.phase == Phase.DONE.value:  # hold the last setpoint
            grip = self._done_grip()

        else:  # a phase this class does not own (subclass hook, e.g. JUDGE)
            grip = self._extra_phase(obs, st)

        action = normalized_action_to(
            cur_pos,
            cur_quat,
            self.target_pos,
            self.grasp_quat,
            grip,
            pos_scale_m=self.pos_scale_m,
            rot_scale_rad=self.rot_scale_rad,
        )
        info = {
            "phase": st.phase,
            # Against the target this step will actually chase (APPROACH may have
            # just switched stage), not the previous step's setpoint.
            "pos_err_m": float(np.linalg.norm(self.target_pos - cur_pos)),
            "contact": in_contact,
            "grasp_confirmed": st.grasp_confirmed,
        }
        return action, info

    # ----------------------------------------------------------------- helpers

    def _advance(self, phase: Phase) -> None:
        self.state.phase = phase.value
        self.state.steps_in_phase = 0
        if phase == Phase.APPROACH:
            # Every APPROACH starts a fresh transit (initial grasp, next
            # candidate, or a re-judge of the same one), docs §9-R43.
            self.state.approach_stage = 0

    # ------------------------------------------------------------- approach path

    def _transit_z(self) -> float:
        """TCP height that clears every body on the table by ``transit_clearance_m``.

        The tallest body sets the floor of this height, not the target's own
        top: the hand crosses the whole table, and the open fingers hang below
        the TCP. The tray rim is folded in as well — it can stand above a short
        body even though it is not something the hand grasps.
        """
        assert self._bundle is not None
        bundle = self._bundle
        tallest = max((float(s.height_m) for s in bundle.object_specs), default=0.0)
        z = float(bundle.cfg["table"]["height_m"]) + tallest
        tray = getattr(self, "tray", None)
        if tray is not None:
            z = max(z, float(tray.rim_z))
        return z + float(self.pipe.transit_clearance_m)

    def _act_approach(self, st: ExpertState, cur_pos: np.ndarray, cur_quat: np.ndarray) -> float:
        """Move onto the approach point, optionally via a safe transit height.

        With ``transit_clearance_m <= 0`` this is the original one-target
        APPROACH: the hand goes straight from wherever it is to the point
        ``approach_clearance_m`` above the grasp. That straight line is at table
        height, so any body standing between the start pose and the target is in
        the way (measured: the first descent from the robot's home pose knocked
        a milk carton over on step 3).

        With it > 0 the APPROACH is split into rise / cross / descend, mirroring
        the CARRY transport: no lateral motion ever happens below the transit
        height. Stage checks use the live TCP position, so a start pose that is
        already above the transit height skips straight to the crossing.
        """
        pipe = self.pipe
        approach = self.grasp_pos + np.array([0.0, 0.0, pipe.approach_clearance_m])
        if float(pipe.transit_clearance_m) <= 0.0:
            self.target_pos = approach
            if float(np.linalg.norm(approach - cur_pos)) < pipe.pos_tol_m and float(
                np.linalg.norm(quat_error_rotvec(self.grasp_quat, cur_quat))
            ) < pipe.rot_tol_rad:
                self._advance(Phase.DESCEND)
            return -1.0

        transit_z = self._transit_z()
        if st.approach_stage == 0:
            # Stage 0: straight up at the current xy. The hand is top-down at
            # reset, so this first leg changes no lateral footprint.
            self.target_pos = np.array([cur_pos[0], cur_pos[1], transit_z])
            if float(cur_pos[2]) >= transit_z - float(pipe.lift_tol_m):
                st.approach_stage = 1
        elif st.approach_stage == 1:
            # Stage 1: across at transit height. Orientation is not required
            # here; it settles during the descent instead of stalling the cross.
            self.target_pos = np.array([approach[0], approach[1], transit_z])
            if float(np.linalg.norm(self.target_pos - cur_pos)) < pipe.pos_tol_m:
                st.approach_stage = 2
        else:
            # Stage 2: down onto the approach point, then the usual DESCEND.
            self.target_pos = approach
            if float(np.linalg.norm(approach - cur_pos)) < pipe.pos_tol_m and float(
                np.linalg.norm(quat_error_rotvec(self.grasp_quat, cur_quat))
            ) < pipe.rot_tol_rad:
                self._advance(Phase.DESCEND)
        return -1.0

    def _on_grasped(self) -> Phase:
        """Phase to enter once the pads have closed on the body (subclass hook)."""
        return Phase.LIFT

    def _on_close_timeout(self) -> Phase:
        """Phase when CLOSE never registered contact (subclass hook).

        The base picker has no candidate bookkeeping: it gives up into the
        lift/hold tail, ends in HOLD and the runner scores the episode a
        failure. A machine that judges candidates overrides this to route the
        empty grasp through its unmeasured path instead of hanging in HOLD.
        """
        return Phase.LIFT

    def _done_grip(self) -> float:
        """Grip command held in DONE (subclass hook).

        The base expert ends holding the body, so it keeps squeezing; an expert
        that has already put the body down overrides this to hold the pads open
        instead, so the recorded trajectory does not end on a close that means
        nothing (docs §9-R39).
        """
        return self.pipe.close_extra

    def _extra_phase(self, obs: dict[str, Any], st: ExpertState) -> float:
        """Handle a phase this class does not own (subclass hook).

        Returns the step's grip command; the subclass owns ``target_pos`` and any
        transition out of the phase. No built-in phase should ever land here.
        """
        raise RuntimeError(f"{type(self).__name__} has no handler for phase {st.phase!r}")


class PickByTemperatureExpert(PickExpert):
    """Pick the body the fingertip temperature says is the target.

    The scene holds several visually similar bodies, so the observations before
    touching carry no temperature information at all: the only signal is the
    measured fingertip trace once the pads are in contact. This expert therefore
    acts the way the task demands -- grasp a candidate, keep holding while the
    fingertip's measured temperature is compared with the target interval, then
    either lift it or put it back and try the next candidate.

    The decision is driven **only** by the policy-rate observation, never by the
    scene's ground-truth object temperature. That is what makes the
    demonstrations imitable: what they teach is "read the temperature channel".
    Every touch is recorded (``temperature_log`` -> the episode's
    ``temperature_judge`` attr) and a wrong commit shows up as a failed episode;
    nothing is hidden.

    Every candidate is judged: a match is carried (the subclass decides how), and
    the machine then returns for the bodies it has not felt yet. Nothing is
    force-committed -- a candidate that never answers the target interval is
    released and left on the table even when it is the last one, so the episode
    can end with nothing carried.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        pipeline: PipelineConfig | None = None,
        pos_scale_m: float = 0.05,
        rot_scale_rad: float = 0.5,
    ):
        super().__init__(
            cfg, pipeline=pipeline, pos_scale_m=pos_scale_m, rot_scale_rad=rot_scale_rad
        )
        self.n_obj = 0
        # Body indices in the order this episode tries them. The identity for
        # every task that does not ask for a permutation, so `cand` is the body
        # index it has always been (docs §9-R39).
        self.order: list[int] = []
        self.dt = 0.05
        self.trace_t: list[float] = []
        self.trace_v: list[np.ndarray] = []
        self._bundle: SceneBundle | None = None
        self._rng: np.random.Generator | None = None
        self._t = 0.0

    # ------------------------------------------------------------------- setup

    def reset(self, bundle: SceneBundle, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        self.n_obj = len(bundle.objects)
        self._bundle, self._rng = bundle, rng
        self.dt = 1.0 / float(bundle.cfg["controller"]["control_hz"])
        self.order = self._order_candidates(bundle, rng)

        jitter = self._set_goal(bundle, self.order[0], rng)
        n_fing = len(np.asarray(bundle.finger_link_names).reshape(-1))
        self.state = self._new_state(
            cand=0,
            onset_s=[None] * n_fing,
            temperature_log=[],
            verdict="idle",
            jitter=jitter,
        )
        self.trace_t.clear()
        self.trace_v.clear()
        self._t = 0.0

    def _new_state(self, **kwargs: Any) -> JudgeState:
        """Build this machine's episode state (subclass hook).

        ``PutInTrayExpert`` returns a :class:`PlaceState` here, so ``reset`` has
        one place to build the state instead of building a ``JudgeState`` and
        copying its fields across by hand (the copy silently dropped any field
        added later).
        """
        return JudgeState(**kwargs)

    def _order_candidates(self, bundle: SceneBundle, rng: np.random.Generator) -> list[int]:
        """Body indices in the order to try them (subclass hook).

        The identity, and — because it draws nothing — no change at all to the
        rng stream every existing task's episodes were generated with.
        """
        return list(range(len(bundle.objects)))

    def _cand_body(self, st: JudgeState) -> int:
        """The *body* index of the candidate the state machine is on."""
        if not self.order:
            return int(st.cand)
        return int(self.order[min(st.cand, len(self.order) - 1)])

    # -------------------------------------------------------------------- step

    def _record(self, obs: dict[str, Any]) -> None:
        """Append this step's fingertip trace and latch each finger's contact onset.

        Recorded from the episode's first step: the pre-contact flat part is the
        fit's baseline, and after a RELEASE the tip carries its previous-grasp
        history, which the estimator's own baseline window absorbs. The onset is
        latched per finger (the pads touch at different control steps) as soon as
        the engine reports link contact.
        """
        st = self.state
        tip = np.asarray(obs["temperature"]["fingertip_grid"], dtype=np.float64).reshape(-1)
        link_contact = np.asarray(obs["contact"]["link_contact"], dtype=bool).reshape(-1)
        self.trace_t.append(self._t)
        self.trace_v.append(tip.copy())
        if st.phase in (Phase.DESCEND.value, Phase.CLOSE.value, Phase.JUDGE.value):
            for i in range(min(len(st.onset_s), link_contact.size)):
                if bool(link_contact[i]) and st.onset_s[i] is None:
                    st.onset_s[i] = self._t
        if st.phase == Phase.JUDGE.value:
            st.verdict = "collecting"
        self._t += self.dt

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        self._record(obs)
        action, info = super().act(obs)
        st = self.state
        info["judge_state"] = st.verdict
        info["candidate_index"] = st.cand
        return action, info

    # --------------------------------------------------------------- judge step

    def _on_grasped(self) -> Phase:
        return Phase.JUDGE

    def _on_close_timeout(self) -> Phase:
        """A grasp that never touched is unmeasured, not a lift.

        JUDGE's all-onsets-missing branch records ``reason="unmeasured"`` and
        releases, so the candidate machine moves on exactly as for any other
        candidate that could not be judged. A timeout implies no link contact,
        hence every ``onset_s`` is still None and that branch is the one taken.
        """
        return Phase.JUDGE

    def _on_positive_verdict(self) -> Phase:
        """Phase after a grasp is judged to be a satisfying body.

        Splitting this out is what lets :class:`PutInTrayExpert` keep the whole
        judge-and-retry machine and only change what happens once the right body
        is in the pads (docs §9-R39).
        """
        return Phase.LIFT

    def _done_grip(self) -> float:
        """End with the pads open: this machine only reaches DONE after a release."""
        return self.pipe.release_grip

    def _extra_phase(self, obs: dict[str, Any], st: JudgeState) -> float:
        if st.phase == Phase.JUDGE.value:
            return self._act_judge(obs, st)
        if st.phase == Phase.RELEASE.value:
            return self._act_release(obs, st)
        return super()._extra_phase(obs, st)

    def _act_judge(self, obs: dict[str, Any], st: JudgeState) -> float:
        """Hold the grasp until the measured tip answers, then branch."""
        return self._act_temperature_judge(obs, st)

    def _act_temperature_judge(self, obs: dict[str, Any], st: JudgeState) -> float:
        """Hold a touched candidate until the measured tip answers, then branch.

        Event-driven, with no fixed window (docs §9-R42; this supersedes the
        first version's 1.5 s hold): every policy step, in this order,

        * any touching finger reads inside the target interval and its trend
          keeps it there -> carry now;
        * no touching finger's trend can still reach that interval -> release
          now;
        * otherwise keep holding and keep reading.

        This estimates no material property: it reads the same measured channel a
        policy sees. Only the configured candidates are in ``self.order``, so a
        rejection costs one touch and never a lift -- the body is released from
        the same pose it was touched in.

        Every candidate is judged: a match is carried, and after the placement
        the machine returns for the candidates still unjudged; a body that does
        not answer the interval is released and the next candidate is tried.
        Nothing is force-carried: a candidate that was never accepted stays on
        the table, and the runner scores the target in the tray at DONE.
        """
        pipe = self.pipe
        self.target_pos = self.grasp_pos
        if not any(o is not None for o in st.onset_s):
            # Contact never registered: nothing was measured. Re-touch the same
            # body when allowed, otherwise treat it as "not the target" and let
            # the candidate machine move on.
            rec = self._unmeasured_record(st)
            st.verdict = "done"
            if st.retry < int(pipe.retry_on_none):
                st.retry += 1
                st.rejudge = True
                rec["retried"] = True
                rec["committed"] = False
                st.temperature_log.append(rec)
            else:
                rec["committed"] = False
                st.temperature_log.append(rec)
            self._advance(Phase.RELEASE)
            return pipe.close_extra

        ev = self._temperature_evidence()
        if not ev["matched"] and not ev["wrong"]:
            return pipe.close_extra  # still undecided: hold and re-read
        rec = self._temperature_record(ev)
        st.temperature_log.append(rec)
        st.verdict = "done"
        if ev["matched"]:
            rec["committed"] = True
            self._advance(self._on_positive_verdict())
        else:
            rec["committed"] = False
            self._advance(Phase.RELEASE)
        return pipe.close_extra

    def _temperature_evidence(self) -> dict[str, Any]:
        """Read the measured fingertip trace as it stands on this step.

        For every finger that touched:

        * baseline: mean of the last ``temperature_judge_baseline_s`` before its
          own onset (diagnostic only, not used by the decision);
        * reading: mean of the last ``temperature_read_tail_s`` now -- enough to
          damp the channel's 0.05 C noise / 0.1 C quantization;
        * slope: least-squares slope over the last ``temperature_trend_window_s``
          of its *post-onset* samples, ``None`` until that many exist (a finger
          that touched late does not vote on a trend it has not seen).

        ``matched`` is "any finger's reading is inside the interval *and* its
        trend keeps it there for one trend window". The second half matters for a
        one-sided interval: a pad chilled by the previous body climbs through a
        cold interval on its way to a hot body (measured 9.97 C at +52 K/s on the
        55 C twin), and that is a pass-through, not arrival; a finger without a
        slope yet cannot match, because "inside now, direction unknown" is not
        evidence of arrival.

        ``wrong`` is "no touching finger can still arrive": every finger that has
        a slope is either inside the interval on its way out or outside it moving
        the wrong way / too slowly. One finger's "no" never vetoes another
        finger's "not yet" -- a pad that touched late has no slope and must not
        cut a real arrival short. A band target needs a fall above it and a rise
        below it.
        """
        cfg = self.cfg
        pipe = self.pipe
        st = self.state
        t = np.asarray(self.trace_t, dtype=np.float64)
        v = np.asarray(self.trace_v, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        n_tail = max(1, int(round(float(pipe.temperature_read_tail_s) / self.dt)))
        n_base = max(1, int(round(float(pipe.temperature_judge_baseline_s) / self.dt)))
        n_trend = max(2, int(round(float(pipe.temperature_trend_window_s) / self.dt)))
        rate = abs(float(pipe.temperature_trend_min_k_per_s))
        window_s = float(pipe.temperature_trend_window_s)
        lo_t, hi_t = cfg["temperature_judge_threshold_c"]
        kind = target_kind(cfg["temperature_judge_threshold_c"])
        reading = v[-n_tail:].mean(axis=0) if len(v) >= n_tail else v.mean(axis=0)
        per_finger: list[dict[str, Any] | None] = []
        matched = False
        viable = False
        for i, onset in enumerate(st.onset_s):
            if onset is None or i >= v.shape[1]:
                per_finger.append(None)
                continue
            i0 = int(np.searchsorted(t, float(onset) - 1e-9))
            i0 = min(max(i0, 0), len(v))
            lo = max(0, i0 - n_base)
            baseline = float(v[lo:i0, i].mean()) if i0 > lo else float(v[min(i0, len(v) - 1), i])
            measured = float(reading[i])
            delta = measured - baseline
            slope: float | None = None
            if len(v) - i0 >= n_trend:
                seg_t = t[-n_trend:].astype(np.float64)
                seg_v = v[-n_trend:, i].astype(np.float64)
                dt_seg = seg_t - seg_t.mean()
                denom = float((dt_seg * dt_seg).sum())
                if denom > 0.0:
                    slope = float((dt_seg * (seg_v - seg_v.mean())).sum() / denom)
            in_interval = bool(
                (lo_t is None or measured >= lo_t) and (hi_t is None or measured <= hi_t)
            )
            if in_interval:
                # Arrival needs the trend to keep the reading inside for one
                # window; a reading on its way out is a crossing, not a match.
                trend_ok: bool | None = None
                if slope is not None:
                    projected = measured + slope * window_s * 2
                    trend_ok = bool(
                        (lo_t is None or projected >= lo_t)
                        and (hi_t is None or projected <= hi_t)
                    )
            elif slope is None:
                trend_ok = None
            elif kind == "cold":
                trend_ok = bool(slope <= -rate)
            elif kind == "hot":
                trend_ok = bool(slope >= rate)
            elif measured > hi_t:
                trend_ok = bool(slope <= -rate)
            else:  # a two-sided band with the reading below its lower bound
                trend_ok = bool(slope >= rate)
            matched = matched or (in_interval and trend_ok is True)
            # "not yet" (None) keeps the finger alive: only a measured trend
            # that cannot reach the interval counts against it.
            viable = viable or trend_ok is not False
            per_finger.append(
                {
                    "onset_s": float(onset),
                    "baseline_c": baseline,
                    "measured_c": measured,
                    "delta_c": delta,
                    "slope_c_per_s": slope,
                    "in_interval": in_interval,
                    "trend_ok": trend_ok,
                }
            )
        return {
            "kind": kind,
            "per_finger": per_finger,
            "matched": bool(matched),
            # Unanimous no from every finger that has a slope; a single finger's
            # rejection is not enough while another can still arrive.
            "wrong": bool((not matched) and (not viable)),
            "threshold_c": [lo_t, hi_t],
            "hold_s": float(t[-1] - max(o for o in st.onset_s if o is not None)),
        }

    def _temperature_record(self, ev: dict[str, Any]) -> dict[str, Any]:
        """Flatten one decision's evidence into the per-touch record."""
        pipe = self.pipe
        st = self.state
        per_finger = ev["per_finger"]
        return {
            "cand": int(st.cand),
            "body": int(self._cand_body(st)),
            "verdict_is_target": bool(ev["matched"]),
            "reason": "in_interval" if ev["matched"] else "wrong_trend",
            "target_kind": ev["kind"],
            "measured_c": [None if f is None else f["measured_c"] for f in per_finger],
            "baseline_c": [None if f is None else f["baseline_c"] for f in per_finger],
            "delta_c": [None if f is None else f["delta_c"] for f in per_finger],
            "slope_c_per_s": [None if f is None else f["slope_c_per_s"] for f in per_finger],
            "per_finger": per_finger,
            "threshold_c": ev["threshold_c"],
            "hold_s": ev["hold_s"],
            "trend_window_s": float(pipe.temperature_trend_window_s),
            "trend_min_k_per_s": float(pipe.temperature_trend_min_k_per_s),
            "baseline_s": float(pipe.temperature_judge_baseline_s),
            "read_tail_s": float(pipe.temperature_read_tail_s),
            "onsets_s": list(st.onset_s),
            "step": int(len(self.trace_t)),
        }

    def _unmeasured_record(self, st: JudgeState) -> dict[str, Any]:
        """The per-touch record for a grasp whose contact never registered."""
        cfg = self.cfg
        pipe = self.pipe
        thresholds = cfg["temperature_judge_threshold_c"]
        return {
            "cand": int(st.cand),
            "body": int(self._cand_body(st)),
            "verdict_is_target": None,
            "reason": "unmeasured",
            "measured_c": None,
            "baseline_c": None,
            "delta_c": None,
            "slope_c_per_s": None,
            "threshold_c": [None if v is None else float(v) for v in thresholds],
            "target_kind": target_kind(thresholds),
            "hold_s": None,
            "trend_window_s": float(pipe.temperature_trend_window_s),
            "trend_min_k_per_s": float(pipe.temperature_trend_min_k_per_s),
            "baseline_s": float(pipe.temperature_judge_baseline_s),
            "read_tail_s": float(pipe.temperature_read_tail_s),
            "onsets_s": list(st.onset_s),
            "unmeasured": True,
            "step": int(len(self.trace_t)),
        }

    def _act_release(self, obs: dict[str, Any], st: JudgeState) -> float:
        """Open the pads, clear the body, then re-aim at the next candidate.

        When nothing is left to judge, the rejected body stays on the table and
        the episode ends -- a candidate that never answered the interval is
        never carried.
        """
        pipe = self.pipe
        self.target_pos = self.grasp_pos + np.array([0.0, 0.0, pipe.approach_clearance_m])
        cur_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        cur_quat = rotvec_to_quat(np.asarray(obs["ee_ori"], dtype=np.float64))
        pos_err = float(np.linalg.norm(self.target_pos - cur_pos))
        rot_err = float(np.linalg.norm(quat_error_rotvec(self.grasp_quat, cur_quat)))
        grip = float(np.asarray(obs["gripper_states"], dtype=np.float64).reshape(-1)[0])
        if grip <= pipe.release_open_tol and pos_err < pipe.pos_tol_m and rot_err < pipe.rot_tol_rad:
            if st.rejudge:
                st.rejudge = False  # same candidate again; st.retry stays spent
                self._reset_touch(st)
                assert self._bundle is not None and self._rng is not None
                st.jitter = self._set_goal(self._bundle, self._cand_body(st), self._rng)
                self._advance(Phase.APPROACH)
            elif self._has_next_candidate(st):
                self._begin_next_candidate(st)
            else:
                self._advance(Phase.DONE)
        return pipe.release_grip

    # ------------------------------------------------- candidate bookkeeping

    def _has_next_candidate(self, st: JudgeState) -> bool:
        """True when another candidate still has to be judged this episode."""
        if not bool(self.cfg["temperature_judge_enabled"]):
            return False
        return st.cand + 1 < (len(self.order) or self.n_obj)

    def _reset_touch(self, st: JudgeState) -> None:
        """Clear the per-touch latches before the next approach.

        The place tail's bookkeeping goes too: whatever it recorded belongs to
        the body just handled, never to the body about to be approached.
        """
        st.onset_s = [None] * len(st.onset_s)
        st.contact_seen = False
        st.squeeze_count = 0
        st.grasp_confirmed = False
        st.verdict = "idle"
        if hasattr(st, "stage"):
            st.stage = 0
        if hasattr(st, "place"):
            st.place = {}

    def _begin_next_candidate(self, st: JudgeState) -> None:
        """Advance ``cand`` and aim at the next unjudged body."""
        st.cand += 1
        st.retry = 0
        self._reset_touch(st)
        assert self._bundle is not None and self._rng is not None
        st.jitter = self._set_goal(self._bundle, self._cand_body(st), self._rng)
        self._advance(Phase.APPROACH)


class PutInTrayExpert(PickByTemperatureExpert):
    """Pick the target body, then carry it to the tray and put it down.

    Two properties keep the task honest:

    * when the temperature judge is enabled the candidate order is a fresh
      permutation per episode -- with a fixed order, "which body does the expert
      feel first" is a constant, and a policy could learn the answer from the
      *instruction* alone without ever reading the temperature channel.
    * the judge reads the measured fingertip channel only, never the scene's
      ground-truth object temperature, so the demonstrations teach the sensor
      contract rather than the simulator's internals.

    The place half: CARRY is two-stage on purpose -- the body rises to a clearance
    height *above the other bodies* before any lateral motion, so the transport
    cannot clip a body it is not holding. LOWER then puts the body's centre one
    half-height above the tray floor, which is where it rests once the pads open;
    DROP opens them and waits for the fingers to clear; RETREAT rises back out of
    the cavity. Every candidate is judged, so every match is carried: successive
    placements use distinct tray slots (:data:`PLACE_SLOTS`) instead of stacking
    on the centre, and RETREAT hands back to the candidate bookkeeping rather
    than ending the episode. The episode's success is read by the runner from the
    scene, not from here (docs §9-R39).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        pipeline: PipelineConfig | None = None,
        pos_scale_m: float = 0.05,
        rot_scale_rad: float = 0.5,
    ):
        super().__init__(
            cfg, pipeline=pipeline, pos_scale_m=pos_scale_m, rot_scale_rad=rot_scale_rad
        )
        self.tray: Any = None

    # ------------------------------------------------------------------- setup

    def reset(self, bundle: SceneBundle, rng: np.random.Generator | None = None) -> None:
        if bundle.tray is None:
            raise RuntimeError("PutInTrayExpert needs a tray: set `tray.enabled: true` in the config")
        self.tray = bundle.tray
        super().reset(bundle, rng)

    def _new_state(self, **kwargs: Any) -> PlaceState:
        return PlaceState(**kwargs)

    def _order_candidates(self, bundle: SceneBundle, rng: np.random.Generator) -> list[int]:
        """The order to try the bodies in, drawn fresh per episode.

        With the temperature judge on, the candidate order is always a
        permutation: with a fixed first body, "which body does the expert feel
        first" is a constant and a policy could read the answer off the
        instruction instead of the temperature channel. The judge only touches
        the configured candidates (the visually similar bodies), not every object
        on the table. Without the judge there is one candidate that matters --
        the pinned target -- so it goes first and the rest are never reached.
        """
        cfg = self.cfg
        if cfg["temperature_judge_enabled"]:
            cands = [int(i) for i in cfg["temperature_candidate"]]
            if len(cands) > 1:
                cands = [cands[int(i)] for i in rng.permutation(len(cands))]
            return cands
        target = int(bundle.target_index)
        return [target] + [i for i in range(len(bundle.objects)) if i != target]

    def _on_grasped(self) -> Phase:
        """With a temperature judge the grasp must be judged before carrying.

        A plain place task pins its body with ``target_index``, so there is
        nothing to judge and the body goes straight to the tray.
        """
        return Phase.JUDGE if self.cfg["temperature_judge_enabled"] else Phase.CARRY

    def _on_close_timeout(self) -> Phase:
        """Nothing in the pads: let the unmeasured path release it.

        With the judge on this is JUDGE (unmeasured -> release -> next
        candidate). Without it there is nothing to judge, so go straight to
        RELEASE, which ends the episode in DONE and leaves it a failure.
        """
        return Phase.JUDGE if self.cfg["temperature_judge_enabled"] else Phase.RELEASE

    # ------------------------------------------------------------------ phases

    def _on_positive_verdict(self) -> Phase:
        return Phase.CARRY

    def _extra_phase(self, obs: dict[str, Any], st: PlaceState) -> float:
        cur_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        if st.phase == Phase.CARRY.value:
            return self._act_carry(obs, st, cur_pos)
        if st.phase == Phase.LOWER.value:
            return self._act_lower(obs, st, cur_pos)
        if st.phase == Phase.DROP.value:
            return self._act_drop(obs, st)
        if st.phase == Phase.RETREAT.value:
            return self._act_retreat(obs, st, cur_pos)
        return super()._extra_phase(obs, st)

    def _carry_z(self) -> float:
        """TCP height for the transport: clear of the tray rim *and* of every body.

        The body hangs one half-height below the TCP, so the rim is not the only
        thing to clear — a body standing on the table is as tall as its own side.
        """
        assert self.tray is not None and self._bundle is not None
        tallest = max((float(s.height_m) for s in self._bundle.object_specs), default=0.0)
        return max(self.tray.rim_z, float(self._bundle.cfg["table"]["height_m"]) + tallest) + float(
            self.pipe.place_carry_clearance_m
        )

    def _slot_xy(self, st: PlaceState) -> np.ndarray:
        """Tray xy for the placement this candidate is about to make.

        The slot is chosen by how many bodies have already been placed, not by
        the candidate index: rejections place nothing, so the first match always
        lands on the centre no matter how many bodies were tried before it.
        """
        assert self.tray is not None
        dx, dy = PLACE_SLOTS[min(len(st.places), len(PLACE_SLOTS) - 1)]
        return np.array(
            [float(self.tray.center_xy[0]) + dx, float(self.tray.center_xy[1]) + dy]
        )

    def _act_carry(self, obs: dict[str, Any], st: PlaceState, cur_pos: np.ndarray) -> float:
        """Stage 0: straight up. Stage 1: across to the tray slot. Then LOWER."""
        pipe = self.pipe
        assert self.tray is not None
        carry_z = self._carry_z()
        if st.stage == 0:
            # Lift first, translate second: the body must be above everything it
            # could clip before it moves sideways.
            self.target_pos = np.array([cur_pos[0], cur_pos[1], carry_z])
            if cur_pos[2] >= carry_z - pipe.lift_tol_m:
                st.stage = 1
            return pipe.close_extra
        slot_xy = self._slot_xy(st)
        self.target_pos = np.array([slot_xy[0], slot_xy[1], carry_z])
        # Cross the table in bounded steps: setting the tray as the target
        # saturates the action at `pos_scale_m` (~1 m/s), and the resulting
        # acceleration shears a can out of the pads (docs §9-R39). The waypoint
        # is still checked against the real goal, not the intermediate point.
        goal = self.target_pos.copy()
        delta = goal - cur_pos
        dist = float(np.linalg.norm(delta))
        if dist > float(pipe.place_carry_step_m):
            self.target_pos = cur_pos + delta * (float(pipe.place_carry_step_m) / dist)
        if dist < pipe.pos_tol_m:
            self._advance(Phase.LOWER)
        return pipe.close_extra

    def _rest_z(self) -> float:
        """TCP height that leaves the held body resting on the tray floor."""
        assert self.tray is not None
        assert self._bundle is not None
        spec = self._bundle.object_specs[self._cand_body(self.state)]
        # The grasp is top-down, so the body's vertical extent is its own height
        # whatever its yaw and the TCP sits at its centre.
        return float(self.tray.floor_z) + 0.5 * float(spec.height_m)

    def _act_lower(self, obs: dict[str, Any], st: PlaceState, cur_pos: np.ndarray) -> float:
        pipe = self.pipe
        assert self.tray is not None
        rest_z = self._rest_z()
        slot_xy = self._slot_xy(st)
        self.target_pos = np.array([slot_xy[0], slot_xy[1], rest_z])
        if float(np.linalg.norm(self.target_pos - cur_pos)) < pipe.place_descend_tol_m:
            if not st.place:
                assert self._bundle is not None
                spec = self._bundle.object_specs[self._cand_body(st)]
                st.place = {
                    "body": self._cand_body(st),
                    "material": str(spec.material),
                    "tray_center_xy": [float(self.tray.center_xy[0]), float(self.tray.center_xy[1])],
                    "tray_slot_xy": [float(slot_xy[0]), float(slot_xy[1])],
                    "tray_floor_z": float(self.tray.floor_z),
                    "rest_z": float(rest_z),
                    "lower_steps": int(st.steps_in_phase),
                }
                st.places.append(dict(st.place))
            self._advance(Phase.DROP)
        return pipe.close_extra

    def _act_drop(self, obs: dict[str, Any], st: PlaceState) -> float:
        """Open the pads and wait for them to clear the body before rising."""
        pipe = self.pipe
        grip = float(np.asarray(obs["gripper_states"], dtype=np.float64).reshape(-1)[0])
        if grip <= pipe.release_open_tol and st.steps_in_phase >= pipe.place_settle_steps:
            self._advance(Phase.RETREAT)
        return pipe.release_grip

    def _act_retreat(self, obs: dict[str, Any], st: PlaceState, cur_pos: np.ndarray) -> float:
        pipe = self.pipe
        assert self.tray is not None
        slot_xy = st.place.get("tray_slot_xy")
        if slot_xy is None:
            slot_xy = [float(self.tray.center_xy[0]), float(self.tray.center_xy[1])]
        self.target_pos = np.array([slot_xy[0], slot_xy[1], self._carry_z()])
        if float(np.linalg.norm(self.target_pos - cur_pos)) < pipe.pos_tol_m:
            # Every candidate is judged: the machine goes back for the bodies it
            # has not felt yet, and only the last one ends the episode.
            if self._has_next_candidate(st):
                self._begin_next_candidate(st)
            else:
                self._advance(Phase.DONE)
        return pipe.release_grip


APPROACHES: dict[str, type[PickExpert]] = {
    "place_in_tray": PutInTrayExpert,
}


def build_expert(
    cfg: dict[str, Any],
    *,
    pipeline: PipelineConfig | None = None,
    pos_scale_m: float = 0.05,
    rot_scale_rad: float = 0.5,
) -> PickExpert:
    """Instantiate the expert named by ``cfg["approach"]``.

    The task YAML names the *goal*, not a class: ``approach: place_in_tray``
    selects :class:`PutInTrayExpert`. New goals register in :data:`APPROACHES`;
    an unknown string is a config error rather than a silent fallback to the
    plain picker (which would ignore the temperature switches).
    """
    name = str(cfg["approach"])
    try:
        cls = APPROACHES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported expert approach {name!r}; supported: {sorted(APPROACHES)}"
        ) from exc
    return cls(cfg, pipeline=pipeline, pos_scale_m=pos_scale_m, rot_scale_rad=rot_scale_rad)


__all__ = [
    "APPROACHES",
    "Phase",
    "PickExpert",
    "PickByTemperatureExpert",
    "PutInTrayExpert",
    "build_expert",
    "ExpertState",
    "JudgeState",
    "PlaceState",
    "TOP_DOWN_QUAT",
]
