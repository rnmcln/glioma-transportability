"""Discrimination, calibration and overall-accuracy metrics for survival models.

For external validation the inverse-probability-of-censoring weights are derived
from the evaluation cohort's own censoring distribution; the development cohort
is not used to weight external metrics. Metrics that cannot be evaluated at a
requested horizon (for example a horizon beyond the last observed event time in
a small subgroup) return NaN rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass

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

from .utils import GLOBAL_SEED, HORIZONS_DAYS, get_logger

log = get_logger("validation")


def _surv(event, time):
    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time).astype(float))


def harrell_c(event, time, risk) -> float:
    return float(concordance_index_censored(np.asarray(event).astype(bool),
                                             np.asarray(time).astype(float),
                                             np.asarray(risk).ravel())[0])


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


def integrated_brier(event, time, surv_prob_matrix, times_days) -> float:
    """surv_prob_matrix: (n, len(times_days)) predicted S(t). Returns IBS."""
    y = _surv(event, time)
    ev = np.asarray(event).astype(bool)
    tmax = np.asarray(time)[ev].max() if ev.any() else np.asarray(time).max()
    keep = [i for i, t in enumerate(times_days) if t < tmax]
    if len(keep) < 2:
        return float("nan")
    times = [times_days[i] for i in keep]
    probs = np.asarray(surv_prob_matrix)[:, keep]
    try:
        return float(integrated_brier_score(y, y, probs, times))
    except Exception:
        return float("nan")


def km_survival_at(event, time, t) -> float:
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(time).astype(float), np.asarray(event).astype(int))
    try:
        return float(kmf.predict(t))
    except Exception:
        return float("nan")


def calibration_in_the_large(event, time, surv_prob_at_t, t) -> float:
    """Observed KM S(t) minus mean predicted S(t). 0 = mean-calibrated."""
    obs = km_survival_at(event, time, t)
    pred = float(np.mean(surv_prob_at_t))
    return obs - pred


def _cox_univariate_slope(event, time, x, max_iter=50, tol=1e-6) -> float:
    """Maximum partial-likelihood estimate of the coefficient of a single
    covariate in a Cox model (Breslow ties), by Newton-Raphson in numpy.

    Equivalent to a univariate ``CoxPHFitter`` but without DataFrame overhead,
    so it is fast enough to run inside a bootstrap loop.
    """
    x = np.asarray(x, float)
    t = np.asarray(time, float)
    e = np.asarray(event).astype(bool)
    order = np.argsort(-t)  # descending time -> cumulative sums give risk sets
    x_s, t_s, e_s = x[order], t[order], e[order]
    b = 0.0
    for _ in range(max_iter):
        w = np.exp(b * x_s)
        S0 = np.cumsum(w)
        S1 = np.cumsum(w * x_s)
        S2 = np.cumsum(w * x_s * x_s)
        # Risk set for each subject = all with time >= t (cumulative under desc sort).
        # Breslow: aggregate ties by taking the last cumulative value per distinct time.
        # Map each event to its risk-set summary at the largest index sharing its time.
        uniq_t, last_idx = np.unique(t_s[::-1], return_index=True)
        # position in descending array of the last (smallest-index) occurrence per time
        last_pos = len(t_s) - 1 - last_idx
        risk_at = {tt: pos for tt, pos in zip(uniq_t, last_pos)}
        ev_idx = np.where(e_s)[0]
        if ev_idx.size == 0:
            return float("nan")
        pos = np.array([risk_at[t_s[i]] for i in ev_idx])
        s0, s1, s2 = S0[pos], S1[pos], S2[pos]
        mean = s1 / s0
        U = np.sum(x_s[ev_idx] - mean)
        I = np.sum(s2 / s0 - mean ** 2)
        if I <= 0 or not np.isfinite(I):
            return float("nan")
        step = U / I
        b += step
        if abs(step) < tol:
            break
    return float(b)


def calibration_slope(event, time, risk) -> float:
    """Slope of the standardised prognostic index in a Cox model fitted in the
    evaluation cohort; 1.0 indicates the spread of predicted risk matches the
    observed risk gradient.
    """
    r = np.asarray(risk).ravel()
    r = (r - r.mean()) / (r.std() or 1.0)
    try:
        return _cox_univariate_slope(event, time, r)
    except Exception:
        return float("nan")


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
        obs = km_survival_at(ev[i], tm[i], t)
        rows.append(dict(bin=str(b), n=len(i), pred=float(np.mean(p[i])), obs=obs))
    return pd.DataFrame(rows)


@dataclass
class EvalResult:
    metrics: dict
    calibration: dict  # horizon_day -> calibration_table DataFrame


def evaluate_model(model, df, horizons_days=HORIZONS_DAYS, primary_horizon=None) -> EvalResult:
    """Compute the full metric panel for a fitted model on an evaluation cohort."""
    event = df["os_event"].to_numpy().astype(bool)
    time = df["os_time_days"].to_numpy().astype(float)
    risk = model.risk_score(df)
    surv = model.survival_at(df, list(horizons_days))  # (n, n_horizons)

    m = {}
    m["harrell_c"] = harrell_c(event, time, risk)
    m["uno_c"] = uno_c(event, time, risk)
    m["ibs"] = integrated_brier(event, time, surv, list(horizons_days))
    m["calibration_slope"] = calibration_slope(event, time, risk)
    auc = time_auc(event, time, risk, list(horizons_days))
    calib = {}
    for j, t in enumerate(horizons_days):
        m[f"auc_{int(round(t/30.4375))}m"] = auc.get(int(t), float("nan"))
        m[f"citl_{int(round(t/30.4375))}m"] = calibration_in_the_large(event, time, surv[:, j], t)
        m[f"km_obs_{int(round(t/30.4375))}m"] = km_survival_at(event, time, t)
        m[f"pred_mean_{int(round(t/30.4375))}m"] = float(np.mean(surv[:, j]))
        calib[int(t)] = calibration_table(event, time, surv[:, j], t)
    return EvalResult(metrics=m, calibration=calib)


def evaluate_arrays(event, time, risk, surv, horizons_days=HORIZONS_DAYS) -> dict:
    """Scalar metric panel from precomputed per-patient risk and S(t).

    Used for the clustered bootstrap: because every model is frozen, per-patient
    predictions are fixed and only the evaluation sample is resampled, so model
    prediction is never repeated inside the bootstrap loop.
    """
    event = np.asarray(event).astype(bool); time = np.asarray(time).astype(float)
    risk = np.asarray(risk).ravel(); surv = np.asarray(surv)
    m = {}
    m["harrell_c"] = harrell_c(event, time, risk)
    m["uno_c"] = uno_c(event, time, risk)
    m["ibs"] = integrated_brier(event, time, surv, list(horizons_days))
    m["calibration_slope"] = calibration_slope(event, time, risk)
    auc = time_auc(event, time, risk, list(horizons_days))
    for j, t in enumerate(horizons_days):
        mo = int(round(t / 30.4375))
        m[f"auc_{mo}m"] = auc.get(int(t), float("nan"))
        m[f"citl_{mo}m"] = calibration_in_the_large(event, time, surv[:, j], t)
        m[f"km_obs_{mo}m"] = km_survival_at(event, time, t)
        m[f"pred_mean_{mo}m"] = float(np.mean(surv[:, j]))
    return m


def bootstrap_arrays(event, time, risk, surv, clusters, n_boot=500,
                     horizons_days=HORIZONS_DAYS, seed=GLOBAL_SEED) -> pd.DataFrame:
    """Cluster bootstrap (resample patients) over precomputed arrays."""
    rng = np.random.default_rng(seed)
    event = np.asarray(event); time = np.asarray(time)
    risk = np.asarray(risk).ravel(); surv = np.asarray(surv)
    clusters = np.asarray(clusters)
    base = evaluate_arrays(event, time, risk, surv, horizons_days)
    keys = list(base.keys())
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in pick])
        try:
            mm = evaluate_arrays(event[idx], time[idx], risk[idx], surv[idx], horizons_days)
        except Exception:
            mm = {k: float("nan") for k in keys}
        for k in keys:
            boots[k].append(mm.get(k, float("nan")))
    rows = []
    for k in keys:
        arr = np.asarray(boots[k], float)
        rows.append(dict(metric=k, point=base[k],
                         lo=float(np.nanpercentile(arr, 2.5)),
                         hi=float(np.nanpercentile(arr, 97.5))))
    return pd.DataFrame(rows)


def bootstrap_metrics(model, df, n_boot=500, horizons_days=HORIZONS_DAYS,
                      cluster_col="patient_id", seed=GLOBAL_SEED) -> pd.DataFrame:
    """Cluster bootstrap (resample patients) of the scalar metric panel.

    Returns a DataFrame with point, lo (2.5%), hi (97.5%) for each metric.
    """
    rng = np.random.default_rng(seed)
    base = evaluate_model(model, df, horizons_days).metrics
    keys = list(base.keys())
    clusters = df[cluster_col].to_numpy()
    uniq = np.unique(clusters)
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        d = df.iloc[idx]
        try:
            mm = evaluate_model(d_with_model(model, d), d, horizons_days).metrics
        except Exception:
            mm = {k: float("nan") for k in keys}
        for k in keys:
            boots[k].append(mm.get(k, float("nan")))
    rows = []
    for k in keys:
        arr = np.asarray(boots[k], float)
        rows.append(dict(metric=k, point=base[k],
                         lo=float(np.nanpercentile(arr, 2.5)),
                         hi=float(np.nanpercentile(arr, 97.5))))
    return pd.DataFrame(rows)


def d_with_model(model, d):
    """Identity helper: the fitted model is frozen; bootstrap resamples only the
    evaluation cohort, never refits. Present for readability in bootstrap_metrics."""
    return model
