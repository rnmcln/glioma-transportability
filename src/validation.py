"""Discrimination, calibration and overall-accuracy metrics for survival models.

Definitions used throughout
---------------------------
* Harrell C: concordance of the model risk score with observed survival over
  the full follow-up.
* Uno C: inverse-probability-of-censoring weighted concordance truncated at the
  90th percentile of observed event times in the evaluation cohort.
* Time-dependent AUC: cumulative/dynamic AUC at each landmark horizon.
* Integrated Brier score (IBS): Brier score integrated over a monthly grid from
  1 month to the last horizon (36 months), truncated below the last observed
  event time.
* Calibration-in-the-large (CITL) at horizon t: Kaplan-Meier S(t) minus the
  mean predicted S(t). Positive values mean predicted survival is too low.
* Calibration slope at horizon t: coefficient of the complementary log-log of
  predicted survival, log(-log S_hat(t | x)), in a Cox model fitted in the
  evaluation cohort with follow-up administratively censored at t. A value of
  1 indicates that the spread of predicted risk matches the observed risk
  gradient; values below 1 indicate predictions that are too extreme.
  Because the predictor is on the log cumulative-hazard scale for every model
  class, the slope is comparable across Cox, tree-based and stacked models.
* Integrated calibration index (ICI), E50 and E90 at horizon t (Austin, Harrell
  and van Klaveren 2020): the observed risk for each patient is estimated from
  a Cox model of the outcome (censored at t) on a restricted cubic spline (3
  knots) of log(-log S_hat(t | x)); ICI is the mean absolute difference between
  observed and predicted risk at t, E50 and E90 its median and 90th percentile.

For external validation the censoring distribution used by IPCW metrics is
estimated in the evaluation cohort itself. Metrics that cannot be evaluated
(for example a horizon beyond the last event in a small subgroup) return NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from .utils import DAYS_PER_MONTH, GLOBAL_SEED, HORIZONS_DAYS, PRIMARY_HORIZON_DAYS, get_logger

log = get_logger("validation")

# Monthly grid used for the integrated Brier score (1 to 36 months).
IBS_GRID_DAYS = tuple(float(round(m * DAYS_PER_MONTH)) for m in range(1, 37))
_EPS = 1e-8


def _surv(event, time):
    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time).astype(float))


# --------------------------------------------------------------------------- discrimination
def harrell_c(event, time, risk) -> float:
    return float(concordance_index_censored(np.asarray(event).astype(bool),
                                             np.asarray(time).astype(float),
                                             np.asarray(risk).ravel())[0])


def harrell_c_pairs(event, time, risk, groups):
    """Harrell C partitioned by whether a comparable pair lies within one group
    or spans two groups.

    Returns a dict with the overall C, C over within-group pairs, C over
    between-group pairs and the number of comparable pairs of each kind.
    Comparable pair: the subject with the shorter time had an event. Ties in
    risk count one half; pairs tied in time are not comparable.
    """
    e = np.asarray(event).astype(bool)
    t = np.asarray(time).astype(float)
    r = np.asarray(risk).astype(float).ravel()
    g = np.asarray(groups).astype(str)
    conc_w = disc_w = tie_w = 0.0
    conc_b = disc_b = tie_b = 0.0
    idx_ev = np.where(e)[0]
    for i in idx_ev:
        later = t > t[i]
        if not later.any():
            continue
        ri, rj = r[i], r[later]
        same = g[later] == g[i]
        c = rj < ri
        d = rj > ri
        tt = rj == ri
        conc_w += c[same].sum(); disc_w += d[same].sum(); tie_w += tt[same].sum()
        conc_b += c[~same].sum(); disc_b += d[~same].sum(); tie_b += tt[~same].sum()

    def _c(cn, dc, ti):
        n = cn + dc + ti
        return float((cn + 0.5 * ti) / n) if n else float("nan"), int(n)

    cw, nw = _c(conc_w, disc_w, tie_w)
    cb, nb = _c(conc_b, disc_b, tie_b)
    call, nall = _c(conc_w + conc_b, disc_w + disc_b, tie_w + tie_b)
    return dict(c_overall=call, c_within=cw, c_between=cb, pairs_within=nw,
                pairs_between=nb, share_between=nb / nall if nall else float("nan"))


def uno_c(event, time, risk, tau=None) -> float:
    y = _surv(event, time)
    try:
        if tau is None:
            tau = np.quantile(np.asarray(time)[np.asarray(event).astype(bool)], 0.9)
        return float(concordance_index_ipcw(y, y, np.asarray(risk).ravel(), tau=tau)[0])
    except Exception:
        return float("nan")


def time_auc(event, time, risk, times_days) -> dict:
    y = _surv(event, time)
    ev = np.asarray(event).astype(bool)
    tmax = np.asarray(time)[ev].max() if ev.any() else np.asarray(time).max()
    times = [t for t in times_days if t < tmax]
    out = {int(t): float("nan") for t in times_days}
    if not times:
        return out
    try:
        aucs, _ = cumulative_dynamic_auc(y, y, np.asarray(risk).ravel(), times)
        for t, a in zip(times, np.atleast_1d(aucs)):
            out[int(t)] = float(a)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- overall accuracy
def integrated_brier(event, time, surv_grid, grid_days=IBS_GRID_DAYS) -> float:
    """surv_grid: (n, len(grid_days)) predicted S(t) on the IBS grid."""
    y = _surv(event, time)
    ev = np.asarray(event).astype(bool)
    tm = np.asarray(time).astype(float)
    tmax = min(tm[ev].max() if ev.any() else tm.max(), tm.max())
    keep = [i for i, t in enumerate(grid_days) if tm.min() <= t < tmax]
    if len(keep) < 2:
        return float("nan")
    try:
        return float(integrated_brier_score(y, y, np.asarray(surv_grid)[:, keep],
                                            [grid_days[i] for i in keep]))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- calibration
def km_survival_at(event, time, t) -> float:
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time).astype(float), np.asarray(event).astype(int))
    try:
        return float(kmf.predict(t))
    except Exception:
        return float("nan")


def calibration_in_the_large(event, time, surv_prob_at_t, t) -> float:
    """Observed KM S(t) minus mean predicted S(t). 0 = mean-calibrated."""
    return km_survival_at(event, time, t) - float(np.mean(surv_prob_at_t))


def cox_fit(event, time, X, max_iter=60, tol=1e-9, ridge=0.0):
    """Cox partial-likelihood fit (Breslow ties) by Newton-Raphson.

    X: (n, p) covariate matrix. Returns (beta, breslow_times, cumulative
    baseline hazard at those times). A tiny optional ridge term stabilises
    near-separated designs; it is zero unless requested.
    """
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(time):
        X = X.T
    t = np.asarray(time, float)
    e = np.asarray(event).astype(bool)
    n, p = X.shape
    order = np.argsort(-t, kind="mergesort")
    Xs, ts, es = X[order], t[order], e[order]
    # Index of the last row (in descending order) sharing each time value, so
    # cumulative sums up to that index give the risk set {j: t_j >= t_i}.
    _, first_rev = np.unique(ts[::-1], return_index=True)
    uniq_t = np.unique(ts[::-1])
    last_pos_by_t = dict(zip(uniq_t, n - 1 - first_rev))
    ev_idx = np.where(es)[0]
    if ev_idx.size == 0:
        return np.full(p, np.nan), np.array([]), np.array([])
    pos = np.array([last_pos_by_t[ts[i]] for i in ev_idx])
    beta = np.zeros(p)
    for _ in range(max_iter):
        w = np.exp(np.clip(Xs @ beta, -50, 50))
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * Xs, axis=0)
        S2 = np.cumsum(w[:, None, None] * (Xs[:, :, None] * Xs[:, None, :]), axis=0)
        s0, s1, s2 = S0[pos], S1[pos], S2[pos]
        mean = s1 / s0[:, None]
        U = (Xs[ev_idx] - mean).sum(0) - ridge * beta
        I = (s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :]).sum(0) + ridge * np.eye(p)
        try:
            step = np.linalg.solve(I, U)
        except np.linalg.LinAlgError:
            return np.full(p, np.nan), np.array([]), np.array([])
        # Step halving guards against overshoot in flat likelihoods.
        if not np.all(np.isfinite(step)):
            return np.full(p, np.nan), np.array([]), np.array([])
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    # Breslow cumulative baseline hazard.
    w = np.exp(np.clip(X @ beta, -50, 50))
    ev_times = np.unique(t[e])
    o = np.argsort(t)
    t_o, w_o, e_o = t[o], w[o], e[o]
    rev_cum = np.cumsum(w_o[::-1])[::-1]
    H, h = np.zeros(len(ev_times)), 0.0
    for k, tt in enumerate(ev_times):
        d = int(np.sum(e_o & (t_o == tt)))
        h += d / rev_cum[np.searchsorted(t_o, tt, side="left")]
        H[k] = h
    return beta, ev_times, H


def _censor_at(event, time, t):
    e = np.asarray(event).astype(bool) & (np.asarray(time, float) <= t)
    tm = np.minimum(np.asarray(time, float), t)
    return e, tm


def _cloglog(surv):
    s = np.clip(np.asarray(surv, float), _EPS, 1 - _EPS)
    return np.log(-np.log(s))


def calibration_slope_at(event, time, surv_at_t, t) -> float:
    """Calibration slope at horizon t (see module docstring)."""
    e, tm = _censor_at(event, time, t)
    if e.sum() < 5:
        return float("nan")
    lp = _cloglog(surv_at_t)
    beta, _, _ = cox_fit(e, tm, lp[:, None])
    return float(beta[0])


def _rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell), linear term plus k-2 terms."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    nk = len(k)
    scale = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(nk - 2):
        term = (np.maximum(x - k[j], 0) ** 3
                - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / scale)
    return np.column_stack(cols)


def smoothed_calibration(event, time, surv_at_t, t, n_knots=3):
    """Observed risk at t for each patient from a cloglog-spline Cox model.

    Returns (predicted_risk, observed_risk) arrays, or (None, None) if the model
    cannot be fitted.
    """
    e, tm = _censor_at(event, time, t)
    pred_risk = 1.0 - np.clip(np.asarray(surv_at_t, float), _EPS, 1 - _EPS)
    # The spline model has three coefficients; with few events before the
    # horizon its fit is unstable, so the ICI is not estimated.
    if e.sum() < 20:
        return None, None
    x = _cloglog(surv_at_t)
    knots = np.quantile(x, [0.1, 0.5, 0.9] if n_knots == 3 else np.linspace(0.05, 0.95, n_knots))
    if len(np.unique(knots)) < n_knots:
        B = x[:, None]
    else:
        B = _rcs_basis(x, knots)
    beta, ev_times, H = cox_fit(e, tm, B, ridge=1e-6)
    if not np.all(np.isfinite(beta)) or len(ev_times) == 0:
        return None, None
    idx = np.searchsorted(ev_times, t, side="right") - 1
    H0_t = H[idx] if idx >= 0 else 0.0
    lp = B @ beta
    obs_risk = 1.0 - np.exp(-H0_t * np.exp(lp))
    return pred_risk, obs_risk


def ici_at(event, time, surv_at_t, t) -> dict:
    p, o = smoothed_calibration(event, time, surv_at_t, t)
    if p is None:
        return dict(ici=float("nan"), e50=float("nan"), e90=float("nan"))
    d = np.abs(o - p)
    return dict(ici=float(d.mean()), e50=float(np.median(d)), e90=float(np.quantile(d, 0.9)))


def calibration_table(event, time, surv_prob_at_t, t, n_bins=5) -> pd.DataFrame:
    """Observed (KM) vs mean-predicted survival at horizon t, by predicted bin."""
    p = np.asarray(surv_prob_at_t).ravel()
    ev = np.asarray(event).astype(int)
    tm = np.asarray(time).astype(float)
    try:
        bins = pd.qcut(p, q=min(n_bins, len(np.unique(p))), duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for b, idx in pd.Series(range(len(p))).groupby(bins, observed=True):
        i = idx.values
        kmf = KaplanMeierFitter().fit(tm[i], ev[i])
        obs = float(kmf.predict(t))
        ci = kmf.confidence_interval_survival_function_
        lo = float(np.interp(t, ci.index.values, ci.iloc[:, 0].values))
        hi = float(np.interp(t, ci.index.values, ci.iloc[:, 1].values))
        rows.append(dict(bin=str(b), n=len(i), pred=float(np.mean(p[i])), obs=obs, obs_lo=lo, obs_hi=hi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fast shared computations
class EvalContext:
    """Quantities that depend only on the evaluation sample (not on the model).

    Computed once per (re)sample and shared by every model, which makes the
    joint bootstrap tractable. Definitions replicate scikit-survival:
    comparable pairs for Harrell C; the reverse Kaplan-Meier censoring
    distribution G (censoring after events at tied times) for IPCW weights in
    Uno C, cumulative/dynamic AUC and the Brier score.
    """

    def __init__(self, event, time, horizons_days=HORIZONS_DAYS, grid_days=IBS_GRID_DAYS):
        self.e = np.asarray(event).astype(bool)
        self.t = np.asarray(time).astype(float)
        e, t = self.e, self.t
        self.n = len(t)
        # Harrell comparable-pair matrix: row i (event) vs column j.
        tj, ti = t[None, :], t[:, None]
        self.comp = ((tj > ti) | ((tj == ti) & ~e[None, :])) & e[:, None]
        self.n_comp = self.comp.sum()
        # Reverse KM censoring survival G.
        ut, inv = np.unique(t, return_inverse=True)
        n_at = np.array([(t >= u).sum() for u in ut])
        d_ev = np.bincount(inv, weights=e.astype(float), minlength=len(ut))
        d_c = np.bincount(inv, weights=(~e).astype(float), minlength=len(ut))
        risk_c = n_at - d_ev
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.where(d_c > 0, 1 - d_c / risk_c, 1.0)
        self._ut, self._G = ut, np.cumprod(fac)
        # Uno C: IPCW-squared weights for events with t < tau (sksurv convention).
        self.tau = float(np.quantile(t[e], 0.9)) if e.any() else float("nan")
        m = t < self.tau
        Gi = self.G(t)
        with np.errstate(divide="ignore"):
            w = np.where(e & (Gi > 0), 1.0 / Gi, 0.0)
        self.w_event = w
        self.uno_mask = m
        self.comp_uno = self.comp & m[:, None]
        self.uno_w = (w ** 2)[:, None] * self.comp_uno
        self.uno_den = self.uno_w.sum()
        # Kaplan-Meier at horizons.
        self.horizons = list(horizons_days)
        self.km_h = self.km(self.horizons)
        self.tmax_event = t[e].max() if e.any() else t.max()
        # Brier score components on the grid.
        self.grid = np.asarray(grid_days, float)
        keep = (self.grid >= t.min()) & (self.grid < min(self.tmax_event, t.max()))
        self.grid_keep = np.where(keep)[0]
        g = self.grid[self.grid_keep]
        self.case_g = (t[:, None] <= g[None, :]) & e[:, None]
        self.ctrl_g = t[:, None] > g[None, :]
        Gg = self.G(g)
        with np.errstate(divide="ignore"):
            self.w_ctrl_g = np.where(Gg > 0, 1.0 / Gg, 0.0)

    def G(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        idx = np.searchsorted(self._ut, x, side="right") - 1
        return np.where(idx >= 0, self._G[np.clip(idx, 0, None)], 1.0)

    def km(self, times):
        t, e = self.t, self.e
        ut = np.unique(t[e])
        n_at = np.array([(t >= u).sum() for u in ut])
        d = np.array([((t == u) & e).sum() for u in ut])
        S = np.cumprod(1 - d / n_at)
        idx = np.searchsorted(ut, np.asarray(times, float), side="right") - 1
        return np.where(idx >= 0, S[np.clip(idx, 0, None)], 1.0)

    # model-dependent pieces -------------------------------------------------
    def harrell(self, r):
        r = np.asarray(r, float)
        d = r[:, None] - r[None, :]
        tie = np.abs(d) <= 1e-8
        num = (self.comp & (d > 0) & ~tie).sum() + 0.5 * (self.comp & tie).sum()
        return float(num / self.n_comp) if self.n_comp else float("nan")

    def uno(self, r):
        if not self.uno_den:
            return float("nan")
        r = np.asarray(r, float)
        d = r[:, None] - r[None, :]
        tie = np.abs(d) <= 1e-8
        num = (self.uno_w * ((d > 0) & ~tie)).sum() + 0.5 * (self.uno_w * tie).sum()
        return float(num / self.uno_den)

    def auc(self, r, t):
        if t >= self.tmax_event:
            return float("nan")
        r = np.asarray(r, float)
        case = (self.t <= t) & self.e
        ctrl = self.t > t
        if not case.any() or not ctrl.any():
            return float("nan")
        d = r[case][:, None] - r[ctrl][None, :]
        tie = np.abs(d) <= 1e-8
        w = self.w_event[case][:, None]
        num = (w * ((d > 0) & ~tie)).sum() + 0.5 * (w * tie).sum()
        return float(num / (self.w_event[case].sum() * ctrl.sum()))

    def ibs(self, surv_grid):
        k = self.grid_keep
        if len(k) < 2:
            return float("nan")
        S = np.asarray(surv_grid, float)[:, k]
        bs = ((S ** 2) * self.case_g * self.w_event[:, None]
              + ((1 - S) ** 2) * self.ctrl_g * self.w_ctrl_g[None, :]).mean(0)
        g = self.grid[k]
        return float(np.trapezoid(bs, g) / (g[-1] - g[0]))


# --------------------------------------------------------------------------- metric panel
def horizon_label(t) -> str:
    return f"{int(round(t / DAYS_PER_MONTH))}m"


def evaluate_arrays(event, time, risk, surv_h, surv_grid, horizons_days=HORIZONS_DAYS,
                    with_ici="all", ctx: EvalContext | None = None) -> dict:
    """Scalar metric panel from per-patient risk scores and predicted survival.

    surv_h: (n, len(horizons_days)) predicted S(t) at the landmark horizons.
    surv_grid: (n, len(IBS_GRID_DAYS)) predicted S(t) on the IBS grid.
    with_ici: "all" (every horizon), "primary" (primary horizon only) or "none".
    """
    if ctx is None:
        ctx = EvalContext(event, time, horizons_days)
    risk = np.asarray(risk).ravel()
    surv_h = np.asarray(surv_h)
    m = {"harrell_c": ctx.harrell(risk), "uno_c": ctx.uno(risk), "ibs": ctx.ibs(surv_grid)}
    for j, t in enumerate(horizons_days):
        h = horizon_label(t)
        m[f"auc_{h}"] = ctx.auc(risk, t)
        m[f"km_obs_{h}"] = float(ctx.km_h[j])
        m[f"pred_mean_{h}"] = float(np.mean(surv_h[:, j]))
        m[f"citl_{h}"] = m[f"km_obs_{h}"] - m[f"pred_mean_{h}"]
        m[f"slope_{h}"] = calibration_slope_at(ctx.e, ctx.t, surv_h[:, j], t)
        if with_ici == "all" or (with_ici == "primary" and t == PRIMARY_HORIZON_DAYS):
            ic = ici_at(ctx.e, ctx.t, surv_h[:, j], t)
            m[f"ici_{h}"] = ic["ici"]; m[f"e50_{h}"] = ic["e50"]; m[f"e90_{h}"] = ic["e90"]
    return m


def bootstrap_panel(event, time, preds: dict, n_boot=500, horizons_days=HORIZONS_DAYS,
                    pairs=(), seed=GLOBAL_SEED, with_ici="primary"):
    """Patient-level bootstrap evaluated jointly for several frozen models.

    preds: {model_name: (risk, surv_h, surv_grid)}. Every model is evaluated on
    the same resampled patients, so differences between models have paired
    bootstrap intervals. ``pairs`` is a sequence of (model, reference) names.
    Returns (summary, diffs): summary has point/lo/hi per model and metric;
    diffs has paired differences (model minus reference) of Harrell C, Uno C,
    IBS and ICI at the primary horizon.
    """
    rng = np.random.default_rng(seed)
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    n = len(event)
    names = list(preds)
    ctx0 = EvalContext(event, time, horizons_days)
    base = {k: evaluate_arrays(event, time, *preds[k], horizons_days, "all", ctx0) for k in names}
    keys = list(base[names[0]].keys())
    boots = {k: {m: [] for m in keys} for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ctx = EvalContext(event[idx], time[idx], horizons_days)
        for k in names:
            r, sh, sg = preds[k]
            try:
                mm = evaluate_arrays(event[idx], time[idx], np.asarray(r)[idx], np.asarray(sh)[idx],
                                     np.asarray(sg)[idx], horizons_days, with_ici, ctx)
            except Exception:
                mm = {m: float("nan") for m in keys}
            for m in keys:
                boots[k][m].append(mm.get(m, float("nan")))
    rows = []
    for k in names:
        for m in keys:
            arr = np.asarray(boots[k][m], float)
            rows.append(dict(model=k, metric=m, point=base[k][m],
                             lo=float(np.nanpercentile(arr, 2.5)),
                             hi=float(np.nanpercentile(arr, 97.5))))
    summary = pd.DataFrame(rows)
    ph = horizon_label(PRIMARY_HORIZON_DAYS)
    drows = []
    for k, ref in pairs:
        if k not in names or ref not in names:
            continue
        for m in ["harrell_c", "uno_c", "ibs", f"ici_{ph}"]:
            if m not in keys:
                continue
            d = np.asarray(boots[k][m], float) - np.asarray(boots[ref][m], float)
            drows.append(dict(model=k, reference=ref, metric=m,
                              diff=base[k][m] - base[ref][m],
                              lo=float(np.nanpercentile(d, 2.5)),
                              hi=float(np.nanpercentile(d, 97.5))))
    diffs = pd.DataFrame(drows)
    return summary, diffs


def bootstrap_c_ci(event, time, risks: dict, n_boot=1000, seed=GLOBAL_SEED) -> dict:
    """Percentile bootstrap intervals for Harrell C of several risk scores
    evaluated on the same resamples. Returns {name: (lo, hi)}."""
    rng = np.random.default_rng(seed)
    event = np.asarray(event).astype(bool); time = np.asarray(time, float)
    n = len(event)
    vals = {k: [] for k in risks}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ctx = EvalContext(event[i], time[i])
        for k, r in risks.items():
            vals[k].append(ctx.harrell(np.asarray(r)[i]) if ctx.n_comp else np.nan)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}
