"""Material judgement from the fingertip's measured thermal response to contact.

The engine's contact heat kernel relaxes the fingertip cell toward the object's
contact cell at a rate that depends on the material alone:

    dT_tip/dt = lambda * (T_cell - T_tip),   lambda = k_eff * A^2 / (V^2 rho c)

with ``k_eff = 2*k_pad*k_obj/(k_pad + k_obj)``. Two consequences make this the
right statistic for a material-judging expert:

* ``lambda`` carries **no temperature dependence** — the same threshold judges a
  hot body and a cold one. That is what keeps the task well posed when every
  episode seeds its objects independently (docs §9-R38/R39); the absolute
  temperature of an object says nothing about its material.
* it is read off the **shipped measured channel** (``obs/temperature/fingertip_grid``,
  format 1.5: RC 0.05 s, noise 0.05 C, resolution 0.1 C — docs §9-R36), so the
  judgement lives inside exactly the observation a learned policy gets. The
  sensor's RC lags the two material bands by different *relative* amounts
  (RC/tau 7.6% vs 3.4%), so the decision threshold must be calibrated on
  measured data, never on ground truth.

Two estimators share the same statistic, and a task picks one with
``judge_method``:

* :func:`estimate_coupling` — the shipped one. Grid-searches the unobservable
  asymptote ``T_cell`` and fits the 15-85% band of a window that is long enough
  for the two parameters to separate (``lambda*W >~ 1.4``); R38 measured the
  recovered band ratio collapsing as the window shrank (2.17x at 2.0 s -> 1.19x
  at 1.0 s).
* :func:`estimate_coupling_ratio` — ``T_cell`` is *eliminated* rather than
  estimated, by taking the ratio of the two half-window slopes. That ratio is
  ``exp(lambda*dt)`` whatever the asymptote is, so the same ``lambda`` comes
  out of a far shorter window: measured on the shipped recordings (16
  fingertips, 3 episodes, docs §9-R40) the three bands separate
  iron/balsa 5.0x and nylon/balsa 2.6x at ``W = 1.0 s``, against 5.5x/2.7x for
  the fit at 4.0 s -- and the band midpoints land at 0.455/1.040, close enough
  to the fit's 0.50/1.04 that the same decision arithmetic carries over.

This module is pure numpy on purpose: the scripted expert runs it online from
the policy-rate observation, and the offline analysis scripts run it on a
recorded episode, with no genesis import on either path (``sim/__init__.py`` is
light, so importing this never pulls the engine).

It is deliberately *not* the same estimator as the historical
``render_grasp_thermal_video.py::fit_rise_tau`` (removed in the 2026-10-20
scripts cleanup, docs §9-R46): that one fitted a whole 165 s run and searched
the asymptote beyond the run's own extremum, while this
one fits a fixed short window anchored at the contact onset. Unifying them
would change the already-published R34/R37 time constants.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "estimate_coupling",
    "estimate_coupling_ratio",
    "judge_fingers",
    "decide",
    "decide_bands",
    "rank_groups",
    "DECIDE_KEYS",
    "BAND_KEYS",
]


def rank_groups(
    groups: Sequence[str],
    conductivities: Sequence[float],
    pad_conductivity: float,
) -> list[str]:
    """The distinct group words, ordered slow -> fast by coupling gain ``k_eff``.

    This is the order :func:`decide_bands` takes its bands in, and it is derived
    from the materials rather than written down in a config, so a band list and
    the scene it judges cannot drift apart. ``k_eff = 2 k_pad k / (k_pad + k)``
    saturates in ``k``, so the ranking is monotone in the object's conductivity
    but not proportional to it (docs §9-R38).
    """
    ranked: list[str] = []
    gains = [
        2.0 * float(pad_conductivity) * float(k) / (float(pad_conductivity) + float(k)) if float(k) > 0.0 else 0.0
        for k in conductivities
    ]
    # `sorted` is stable: equal gains keep their first-appearance order, so two
    # bodies of one material never split a band.
    for i in sorted(range(len(groups)), key=lambda i: gains[i]):
        name = str(groups[i])
        if name not in ranked:
            ranked.append(name)
    return ranked


def _as_2d(tips: Any) -> np.ndarray:
    """(T,) | (T, F) | (T, F, 1, 1, 1) -> (T, F)."""
    arr = np.asarray(tips, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        if arr.shape[2] != 1:
            raise ValueError(f"expected one lumped cell per finger, got {arr.shape}")
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"cannot read fingertip trace of shape {np.shape(tips)}")
    return arr


def estimate_coupling(
    t: Any,
    tip: Any,
    t_on: float,
    *,
    window_s: float,
    settle_s: float = 0.15,
    band: tuple[float, float] = (0.15, 0.85),
    tc_hint: float | None = None,
    tc_span_k: float | None = None,
    min_span_k: float = 0.5,
    min_samples: int = 6,
    n_grid: int = 201,
) -> dict[str, Any] | None:
    """First-order fit of one fingertip's relaxation inside one contact window.

    The model is ``T(t) = T_cell + (T_0 - T_cell) * exp(-lambda * (t - t_on))``,
    time origin at the contact onset: the samples inside ``[t_on, t_on +
    settle_s]`` are dropped (the engine's contact query and the sensor RC need a
    few control steps to settle) but the origin stays at ``t_on``, where the
    response physically starts.

    ``T_cell`` is not observable, so it is grid-searched — beyond the window's
    own furthest excursion when ``tc_hint`` is None (the same shape as
    ``fit_rise_tau``'s search), or inside ``tc_hint ± tc_span_k`` when the
    object's own grid channel supplies a seed (a much narrower, better
    conditioned search). Only the 15–85% band of the excursion is fitted, so the
    fit never sees the pre-onset flat part or the far saturated tail.

    Returns ``{lambda_s, tau_s, t_cell_c, v0_c, span_k, sign, r2, n,
    plateau_rel, ...}`` or None when the window holds too few samples or too
    little motion (``abs(span_k) < min_span_k``).

    ``plateau_rel`` is the relative width of the set of grid points whose R^2 is
    within 1e-4 of the best: near 0 for a well-conditioned window, large when
    the trace is nearly linear and ``T_cell``/``lambda`` cannot be separated.
    """
    t = np.asarray(t, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64).reshape(-1)
    if t.shape != tip.shape:
        raise ValueError(f"t {t.shape} and tip {tip.shape} must match")

    t_on = float(t_on)
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"band must be 0 <= lo < hi <= 1, got {band!r}")

    # Baseline: the tip just before the fingers touched.
    base = (t >= t_on - 0.15) & (t <= t_on)
    fit = (t >= t_on + settle_s) & (t <= t_on + settle_s + window_s)
    if int(fit.sum()) < min_samples:
        return None
    v0 = float(tip[base].mean()) if bool(base.any()) else float(tip[fit][0])
    t_f = t[fit]
    v_f = tip[fit]

    delta = v_f - v0
    i_ex = int(np.argmax(np.abs(delta)))
    span = float(delta[i_ex])
    sign = 1.0 if span >= 0.0 else -1.0
    if abs(span) < float(min_span_k):
        return None

    v_ex = float(v_f[i_ex])
    if tc_hint is None:
        grid = v_ex + sign * np.linspace(0.0, 0.5 * abs(span), int(n_grid))
    else:
        half = float(tc_span_k) if tc_span_k is not None else max(0.5 * abs(span), 5.0)
        grid = np.linspace(float(tc_hint) - half, float(tc_hint) + half, int(n_grid))
        # An asymptote on the wrong side of the excursion cannot describe it.
        grid = grid[sign * (grid - v_ex) > 0.0]
    if grid.size < 2:
        return None

    frac = sign * delta / abs(span)
    sel = (frac >= lo) & (frac <= hi)
    if int(sel.sum()) < min_samples:
        return None
    t_s, v_s = t_f[sel], v_f[sel]
    ss_tot = float(((v_s - v_s.mean()) ** 2).sum())

    r2s = np.full(grid.shape, -np.inf)
    lams = np.full(grid.shape, np.nan)
    for i, tc in enumerate(grid):
        resid = np.log(np.maximum(sign * (tc - v_s), 1e-9))
        slope, _ = np.polyfit(t_s, resid, 1)
        if not np.isfinite(slope) or slope >= 0.0:
            continue
        lam = float(-slope)
        pred = tc - (tc - v0) * np.exp(-lam * (t_f - t_on))
        ss_res = float(((v_f - pred) ** 2).sum())
        r2s[i] = 1.0 - ss_res / max(ss_tot, 1e-12)
        lams[i] = lam

    ok = np.isfinite(r2s)
    if not bool(ok.any()):
        return None
    best = float(np.nanmax(r2s))
    i_best = int(np.nanargmax(r2s))
    plateau = ok & (r2s >= best - 1e-4)
    lam_lo = float(np.nanmin(lams[plateau]))
    lam_hi = float(np.nanmax(lams[plateau]))
    lam = float(lams[i_best])
    return {
        "lambda_s": lam,
        "tau_s": (1.0 / lam) if lam > 0.0 else float("inf"),
        "t_cell_c": float(grid[i_best]),
        "v0_c": v0,
        "v_end_c": float(v_f[-1]),
        "span_k": span,
        "sign": sign,
        "r2": best,
        "n": int(fit.sum()),
        "plateau_rel": (lam_hi - lam_lo) / lam if lam > 0.0 else float("inf"),
        "window_s": float(window_s),
        "settle_s": float(settle_s),
        "band": (lo, hi),
    }


def estimate_coupling_ratio(
    t: Any,
    tip: Any,
    t_on: float,
    *,
    window_s: float,
    settle_s: float = 0.15,
    min_span_k: float = 0.5,
    min_samples: int = 10,
    min_half_samples: int = 4,
) -> dict[str, Any] | None:
    """``lambda`` from the ratio of the two half-window slopes, asymptote-free.

    The engine's response to contact has slope ``dT/dt = lambda * (T_cell - T(t))``,
    i.e. ``lambda * D_0 * exp(-lambda * (t - t_on))`` with ``D_0 = T_cell - T(t_on)``.
    ``T_cell`` sets only the *scale* of that slope and cancels in the ratio of two
    half-window slopes:

        s1 / s2 = exp(lambda * (t2 - t1))   ->   lambda = ln(s1 / s2) / (t2 - t1)

    so the parameter that needs the long window in :func:`estimate_coupling` is
    removed instead of estimated, and the window only has to hold two distinguishable
    slopes. The sign of ``D_0`` (hot object under a cold tip, or the reverse after a
    previous grasp left the tip hot) does not matter: both slopes carry it and the
    ratio drops it.

    Measured on the shipped recordings (8 grasps over 3 episodes, docs §9-R40),
    re-judged offline (the historical ``scripts/judge_temperature_material.py``,
    removed in the 2026-10-20 scripts cleanup, docs §9-R46) with the window the
    online verdict actually fired on: wooden 0.2629-0.2704, plastic
    0.7451-0.7581, iron 1.4914-1.4979 at ``W = 1.5`` against the ``fit``'s
    0.2682-0.2704 / 0.7216-0.7400 / 1.4881-1.4912 at ``W = 4.0``. Same three bands,
    same scale, a quarter of the hold.

    ``window_s`` has a floor that is not about conditioning: the two half-window
    slopes need ``max(min_samples, 2*min_half_samples)`` samples between them, and
    when the anchor sits at the earlier pad's touch the *late* pad has already spent
    the inter-pad gap (0.25-0.6 s here) before its window opens. At ``W = 1.0`` a
    600 ms gap leaves it 9 samples against ``min_samples = 10`` and the estimate is
    dropped, taking the two-finger average with it; ``W = 1.5`` clears it by three
    samples. Raise ``window_s`` before lowering these gates.

    Returns the same keys as :func:`estimate_coupling` plus the two half-slopes, so a
    recorded episode can be re-judged offline with either estimator. ``t_cell_c`` is
    recovered as ``v0 + s1/lambda`` (the asymptote the ratio cancelled) and is a
    diagnostic only -- the emitted channel records the raw object grid, and no
    judgement reads it. ``plateau_rel`` does not apply here and is NaN.
    """
    t = np.asarray(t, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64).reshape(-1)
    if t.shape != tip.shape:
        raise ValueError(f"t {t.shape} and tip {tip.shape} must match")

    t_on = float(t_on)
    base = (t >= t_on - 0.15) & (t <= t_on)
    fit = (t >= t_on + settle_s) & (t <= t_on + settle_s + window_s)
    t_f, v_f = t[fit], tip[fit]
    n = int(t_f.size)
    h = n // 2
    if n < max(int(min_samples), 2 * int(min_half_samples)):
        return None
    v0 = float(tip[base].mean()) if bool(base.any()) else float(v_f[0])
    span = float(v_f[int(np.argmax(np.abs(v_f - v0)))] - v0)
    sign = 1.0 if span >= 0.0 else -1.0
    if abs(span) < float(min_span_k):
        return None

    x = t_f - t_on
    (s1, i1) = np.polyfit(x[:h], v_f[:h], 1)
    (s2, i2) = np.polyfit(x[h:], v_f[h:], 1)
    # Same sign, first slope the steeper one: anything else is not a relaxation.
    if s1 == 0.0 or s2 == 0.0 or s1 * s2 <= 0.0 or abs(s1) <= abs(s2):
        return None
    dt_mid = float(x[h:].mean() - x[:h].mean())
    if dt_mid <= 0.0:
        return None
    lam = float(np.log(abs(s1) / abs(s2)) / dt_mid)
    if not np.isfinite(lam) or lam <= 0.0:
        return None

    def _half_r2(xx: np.ndarray, yy: np.ndarray, s: float, i: float) -> float:
        ss_res = float(((yy - (s * xx + i)) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    def _half_se(xx: np.ndarray, yy: np.ndarray, s: float, i: float) -> float:
        """Standard error of this half's slope, from its own residuals."""
        if xx.size < 3:
            return float("nan")
        ss_res = float(((yy - (s * xx + i)) ** 2).sum())
        sxx = float(((xx - xx.mean()) ** 2).sum())
        if sxx <= 0.0:
            return float("nan")
        return float(np.sqrt(max(ss_res, 0.0) / (xx.size - 2) / sxx))

    se1, se2 = _half_se(x[:h], v_f[:h], s1, i1), _half_se(x[h:], v_f[h:], s2, i2)
    # d(ln(s1/s2)) = ds1/s1 - ds2/s2, independent errors: quadrature.
    sigma = float(np.sqrt((se1 / s1) ** 2 + (se2 / s2) ** 2) / dt_mid) if np.isfinite(se1 + se2) else float("nan")
    return {
        "lambda_s": lam,
        "tau_s": 1.0 / lam,
        # The cancelled asymptote, put back for diagnostics only.
        "t_cell_c": float(v0 + s1 / lam) if lam > 0.0 else float("nan"),
        "v0_c": v0,
        "v_end_c": float(v_f[-1]),
        "span_k": span,
        "sign": sign,
        "r2": float(min(_half_r2(x[:h], v_f[:h], s1, i1), _half_r2(x[h:], v_f[h:], s2, i2))),
        "n": n,
        "plateau_rel": float("nan"),
        "window_s": float(window_s),
        "settle_s": float(settle_s),
        "band": None,
        "method": "ratio",
        "slope_first": float(s1),
        "slope_second": float(s2),
        "dt_mid_s": dt_mid,
        "sigma_lambda_s": sigma,
    }


def judge_fingers(
    t: Any,
    tips: Any,
    t_on: float | Sequence[float],
    *,
    window_s: float,
    settle_s: float = 0.15,
    band: tuple[float, float] = (0.15, 0.85),
    tc_hints: Sequence[float | None] | None = None,
    method: str = "fit",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one estimator per fingertip and pool the estimates.

    ``method`` picks it: ``"fit"`` (default, the shipped grid search over the
    asymptote -- :func:`estimate_coupling`) or ``"ratio"`` (asymptote-free,
    :func:`estimate_coupling_ratio`, for tasks that trade the hold for a shorter
    window). Both return the same keys, so a recorded episode re-judges with either.

    ``tips`` may be (T, F), (T, F, 1, 1, 1) or (T,); ``tc_hints`` optionally
    seeds each finger's asymptote search (e.g. from the object's grid channel) and
    is read by the ``"fit"`` path only -- the ratio estimator has no asymptote to
    seed.

    ``t_on`` is per finger and matters: the two pads of one grasp do **not**
    touch at the same control step (measured on the shipped cold renders: tip L
    onset 1.15 s vs tip R 1.40 s). Fitting a late finger from the early finger's
    onset mixes pre-contact flat into its window and biases its lambda low —
    measured here as tau 0.585 s (L) vs 0.954 s (R) against the whole-run fit's
    0.656/0.656 s. A scalar is broadcast to every finger.

    The pooled ``lambda_s`` is the mean over the fingers that produced a fit —
    under the pad-face contact-area policy (docs §9-R37) both fingers of a grasp
    see the same contact area, so a large disagreement is itself a signal that
    the grasp is off (one pad on an edge).
    """
    arr = _as_2d(tips)
    n_f = arr.shape[1]
    hints: list[float | None] = list(tc_hints) if tc_hints is not None else [None] * n_f
    if len(hints) != n_f:
        raise ValueError(f"tc_hints has {len(hints)} entries for {n_f} fingers")
    onsets: list[float | None]
    if t_on is None:
        onsets = [None] * n_f
    elif np.ndim(t_on) == 0:
        onsets = [float(t_on)] * n_f
    else:
        onsets = [None if v is None or not np.isfinite(v) else float(v) for v in t_on]
        if len(onsets) != n_f:
            raise ValueError(f"t_on has {len(onsets)} entries for {n_f} fingers")
    if method not in ("fit", "ratio"):
        raise ValueError(f"unknown judge method {method!r}; expected 'fit' or 'ratio'")
    per = [
        None if onsets[i] is None
        else (
            estimate_coupling_ratio(
                t, arr[:, i], onsets[i], window_s=window_s, settle_s=settle_s, **kwargs,
            )
            if method == "ratio"
            else estimate_coupling(
                t, arr[:, i], onsets[i], window_s=window_s, settle_s=settle_s, band=band,
                tc_hint=hints[i], **kwargs,
            )
        )
        for i in range(n_f)
    ]
    lams = [e["lambda_s"] for e in per if e is not None]
    lam = float(np.mean(lams)) if lams else None
    spread = (
        float((max(lams) - min(lams)) / lam) if len(lams) > 1 and lam else None
    )
    return {"fingers": per, "lambda_s": lam, "n_fingers": len(lams), "spread_rel": spread}


DECIDE_KEYS = ("lambda_s", "threshold", "group", "margin")


def decide(
    lambda_s: float | None,
    threshold: float,
    *,
    groups: tuple[str, str] = ("metal", "nonmetal"),
) -> dict[str, Any]:
    """Threshold decision: fast coupling = the conductive group.

    ``margin`` is ``lambda_s / threshold`` (how many thresholds the measurement
    sits past the decision boundary; <1 means the other group). The expert
    records it so a recorded episode carries its own confidence, and so the
    offline confusion matrix can be swept without re-running the scene.
    """
    if lambda_s is None or not np.isfinite(lambda_s):
        return {"lambda_s": None, "threshold": float(threshold), "group": None, "margin": None}
    fast, slow = groups
    margin = float(lambda_s) / float(threshold) if threshold > 0.0 else float("inf")
    return {
        "lambda_s": float(lambda_s),
        "threshold": float(threshold),
        "group": fast if margin >= 1.0 else slow,
        "margin": margin,
    }


BAND_KEYS = DECIDE_KEYS + ("thresholds", "band", "margins")


def decide_bands(
    lambda_s: float | None,
    thresholds: Sequence[float],
    *,
    groups: Sequence[str],
) -> dict[str, Any]:
    """Banded decision: ``len(thresholds) + 1`` groups ordered slow -> fast.

    ``decide`` splits two groups with one threshold; a task whose bodies come in
    three materials needs two, and the same one number-per-boundary arithmetic
    extends: ``groups[i]`` is the answer while the measurement sits at or above
    ``thresholds[i-1]`` and below ``thresholds[i]``. ``groups`` must therefore be
    ordered by increasing coupling (the caller ranks them by ``k_eff``, which is
    the material's whole contribution to lambda — docs §9-R38), and
    ``thresholds`` must be increasing.

    The returned dict is a superset of :func:`decide`'s: ``group`` is the band's
    own group, ``threshold``/``margin`` are the *nearest* boundary's (the one the
    decision had to clear or miss), and ``band``/``thresholds``/``margins`` carry
    the whole picture. A recorded episode keeps all of it, so the two boundaries
    can be re-swept offline exactly like the single threshold can (docs §9-R39).
    """
    thr = [float(t) for t in thresholds]
    grp = [str(g) for g in groups]
    if not thr:
        raise ValueError("decide_bands needs at least one threshold; use decide() for one boundary")
    if len(grp) != len(thr) + 1:
        raise ValueError(f"{len(thr)} thresholds need {len(thr) + 1} groups, got {len(grp)}: {grp}")
    if any(b <= a for a, b in zip(thr, thr[1:])):
        raise ValueError(f"thresholds must be increasing and distinct, got {thr}")
    if lambda_s is None or not np.isfinite(lambda_s):
        return {
            "lambda_s": None,
            "threshold": None,
            "group": None,
            "margin": None,
            "thresholds": thr,
            "band": None,
            "margins": None,
        }
    lam = float(lambda_s)
    margins = [lam / t if t > 0.0 else float("inf") for t in thr]
    # The band is set by every boundary the measurement has cleared, so this is
    # the same comparison `decide` makes at each boundary, applied n times.
    band = int(sum(1 for m in margins if m >= 1.0))
    # Nearest boundary: the one just above the measurement when it did not clear
    # all of them, else the topmost it did clear (`margin` then reads as "how far
    # past the last boundary", which is the useful confidence for a fast band).
    near = band if band < len(thr) else len(thr) - 1
    return {
        "lambda_s": lam,
        "threshold": thr[near],
        "group": grp[band],
        "margin": margins[near],
        "thresholds": thr,
        "band": band,
        "margins": margins,
    }


def _mean_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0
