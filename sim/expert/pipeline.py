"""Internal constants of the scripted expert pipeline.

The task YAML names *what* the expert should do — an ``approach`` string (the
goal) plus the three temperature-judge switches. Everything the state machine
needs to know *how* to do it lives here: approach/transit geometry, grasp
tolerances, the retired lambda/material judge, the event-driven temperature
judge, the release behaviour and the carry-to-tray tail.

They used to be ``ExpertConfig`` fields (and therefore YAML knobs). Keeping
them in one plain dataclass still lets a development script override them
in-process — ``dataclasses.replace(PIPELINE, ...)`` — without turning them
back into a per-task contract.

``PIPELINE`` is the live default; frozen so a task run cannot mutate it by
accident. Pass a replacement down through ``EpisodeRunner(..., pipeline=...)``
or an expert constructor's ``pipeline=`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass


def target_kind(threshold_interval: tuple[float | None, float | None]) -> str:
    """Classify a temperature target interval: ``cold`` / ``hot`` / ``band``.

    ``[null, hi]`` is a cold target (below ``hi``), ``[lo, null]`` a hot target
    (above ``lo``), and two finite bounds a two-sided band. The expert uses the
    kind only to decide which way a fingertip reading must be moving to reach
    the interval (down for cold, up for hot, toward the nearer bound for a band).
    """
    lo, hi = threshold_interval
    if lo is None and hi is not None:
        return "cold"
    if lo is not None and hi is None:
        return "hot"
    return "band"


@dataclass(frozen=True)
class PipelineConfig:
    """Every expert constant that is not part of the task contract."""

    # ------------------------------------------------ approach / grasp geometry
    # Approach point height above the grasp point (docs §5.6).
    approach_clearance_m: float = 0.08
    # Safe transit (docs §9-R43): rise to table + tallest body + this before any
    # lateral motion. 0 disables the two-stage APPROACH (the pre-R43 straight
    # line at table height, which can knock over a body between home and target).
    transit_clearance_m: float = 0.10
    grasp_z_offset_m: float = 0.0  # TCP target z relative to the object centre

    # Per-episode randomization ranges (widen = more diverse, harder demos).
    grasp_xy_jitter_m: float = 0.004
    approach_yaw_jitter_rad: float = 0.15

    # ------------------------------------------------------ state-machine gates
    pos_tol_m: float = 0.01
    rot_tol_rad: float = 0.2
    descend_tol_m: float = 0.015
    contact_depth_m: float = 1e-5  # probe depth above which contact counts
    squeeze_steps: int = 10  # closed-loop steps kept before lifting/judging
    close_timeout_steps: int = 40  # give up closing after this many steps
    lift_height_m: float = 0.10
    lift_tol_m: float = 0.012
    close_extra: float = 1.0  # grip command in CLOSE/LIFT/HOLD/CARRY (1 = closed)

    # --------------------------------------------------------------- release
    release_grip: float = -1.0  # action-space command (-1 = fully open)
    release_open_tol: float = -0.6  # gripper_states[0] below this counts as open

    # ------------------------------------------ material / lambda judge (R38-R40)
    # Contact time fitted after the settle skip. `fit` needs lambda*W >~ 1.4;
    # `ratio` works at 1.5 s (docs §9-R40).
    judge_window_s: float = 2.0
    judge_method: str = "fit"  # "fit" | "ratio"
    judge_settle_s: float = 0.15  # engine contact query + sensor RC transient
    # 1.0 = anchor every finger's window at the later onset. Lowering it lets
    # the clock start at the earlier touch while the late finger still keeps
    # this fraction of the window.
    judge_min_late_window_frac: float = 1.0
    # lambda = k_eff*A^2/(V^2*rho*c), independent of temperature, so one
    # threshold separates a hot body from a cold one (docs §9-R38).
    judge_lambda_threshold: float = 1.04
    judge_fit_band: tuple[float, float] = (0.15, 0.85)
    # Non-empty -> multi-band judgement: thresholds[i] separates band i from
    # band i+1 in slow -> fast group order (docs §9-R39).
    judge_lambda_thresholds: tuple[float, ...] = ()
    # A fit whose fingertip excursion is below this is rejected as noise
    # dominated (docs §9-R39).
    judge_min_span_k: float = 0.5

    # ------------------------------------------------- temperature judge (R42)
    # Event-driven: there is no fixed hold window. Every policy step, a touching
    # finger whose smoothed reading is inside the target interval ends the hold
    # immediately (carry); a finger whose trend cannot reach the interval ends it
    # just as immediately (release); only "undecided" keeps holding. The measured
    # fingertip response has tau ~0.66 s under pad_face, so the trend is already
    # clear during the 10-step squeeze that precedes JUDGE, and a cold pull
    # usually reaches the interval within a few JUDGE steps.
    #
    # Trend = least-squares slope over this many seconds of the finger's
    # post-onset samples. 0.4 s is 8 policy steps at 20 Hz, enough to put a
    # slope well above the channel's 0.05 C noise / 0.1 C quantization.
    temperature_trend_window_s: float = 0.4
    # A finger is "heading toward the interval" only at this rate or faster in
    # the required direction (K/s). A flat pad is therefore not heading there.
    # The release needs *no* touching finger heading there, so a late-touching
    # pad without a slope yet never vetoes one that is; that is also what rejects
    # the warm twin without waiting for a timeout. The threshold sits above the
    # slope noise (~0.2 K/s) and far below a real cold pull (~2-20 K/s).
    temperature_trend_min_k_per_s: float = 0.25
    # Baseline = mean of this many seconds before the finger's own onset; kept
    # for the per-touch record (diagnostic), not needed by the decision.
    temperature_judge_baseline_s: float = 0.3
    # Reading = mean of the last this-many seconds. This is a smoothing constant,
    # not a hold time: the hold ends on the first step whose smoothed reading
    # satisfies a rule.
    temperature_read_tail_s: float = 0.15

    # ------------------------------- unusable-contact retry (both judges, R39)
    # Re-grasp the *same* candidate up to this many times when the contact was
    # too poor to measure. 0 = advance/commit as R38 did.
    retry_on_none: int = 0

    # --------------------------------------------- put-in-tray tail (R39)
    place_descend_tol_m: float = 0.010  # LOWER -> DROP position tolerance
    place_settle_steps: int = 5  # steps to hold after opening before RETREAT
    # TCP clearance above the tray rim and the tallest body while carrying.
    place_carry_clearance_m: float = 0.10
    # Lateral speed cap for CARRY, metres per policy step. Without it the action
    # saturates at pos_scale_m (~1 m/s) and shears the body out of the pads.
    place_carry_step_m: float = 0.02


PIPELINE = PipelineConfig()

__all__ = ["PIPELINE", "PipelineConfig", "target_kind"]
