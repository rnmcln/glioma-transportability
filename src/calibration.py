"""Recalibration of a frozen model on the log cumulative-hazard scale.

The prognostic index (PI) of a frozen model is defined for every model class
as the complementary log-log of its predicted survival at the primary horizon,
PI = log(-log S_hat(24 months | x)). For a proportional-hazards model this is
the linear predictor plus a constant, so relative hazards are preserved
exactly; for tree-based and stacked models it places the prediction on the
same scale. The PI is centred but never rescaled.

Recalibration re-estimates, in the target cohort, a Breslow baseline
cumulative hazard and optionally a slope on the PI (van Houwelingen 2000):

* ``in_large``: S(t | x) = exp(-H0(t) exp(PI_c)), slope fixed at 1;
* ``slope``:    S(t | x) = exp(-H0(t) exp(b PI_c)), b estimated with follow-up
  administratively censored at the primary horizon, so that it refers to the
  horizon at which calibration is assessed;
* stratified:   a separate H0 per stratum with b estimated from the
  stratified partial likelihood (feature weights unchanged).
"""
from __future__ import annotations

import numpy as np

from .validation import cox_fit

_EPS = 1e-8


def prognostic_index(surv_at_primary):
    s = np.clip(np.asarray(surv_at_primary, float), _EPS, 1 - _EPS)
    return np.log(-np.log(s))


def _censor(event, time, horizon):
    """Administrative censoring at ``horizon`` (used when estimating slopes, so
    that the slope refers to the same horizon as the calibration metrics)."""
    e = np.asarray(event).astype(bool); t = np.asarray(time, float)
    if horizon is None:
        return e, t
    return e & (t <= horizon), np.minimum(t, horizon)


def breslow_baseline(event, time, offset):
    """Breslow cumulative baseline hazard for fixed offset (log relative hazard)."""
    t = np.asarray(time, float)
    e = np.asarray(event).astype(bool)
    w = np.exp(np.asarray(offset, float))
    order = np.argsort(t, kind="mergesort")
    t_s, e_s, w_s = t[order], e[order], w[order]
    rev_cum_w = np.cumsum(w_s[::-1])[::-1]
    uniq = np.unique(t_s[e_s])
    H0 = np.zeros(len(uniq))
    h = 0.0
    for k, tt in enumerate(uniq):
        d = int(np.sum(e_s & (t_s == tt)))
        h += d / rev_cum_w[np.searchsorted(t_s, tt, side="left")]
        H0[k] = h
    return uniq, H0


def _eval_H0(times_, H0_, t):
    t = np.atleast_1d(np.asarray(t, float))
    idx = np.searchsorted(times_, t, side="right") - 1
    return np.where(idx >= 0, H0_[np.clip(idx, 0, max(len(H0_) - 1, 0))], 0.0) if len(H0_) else np.zeros(len(t))


def _stratified_slope(event, time, x, strata, max_iter=50, tol=1e-9):
    """One-covariate Cox coefficient from the stratified partial likelihood."""
    b = 0.0
    e = np.asarray(event).astype(bool); t = np.asarray(time, float); x = np.asarray(x, float)
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for _ in range(max_iter):
        U = I = 0.0
        for g in groups:
            tg, eg, xg = t[g], e[g], x[g]
            o = np.argsort(-tg, kind="mergesort")
            ts, es, xs = tg[o], eg[o], xg[o]
            w = np.exp(b * xs)
            S0, S1, S2 = np.cumsum(w), np.cumsum(w * xs), np.cumsum(w * xs * xs)
            _, first_rev = np.unique(ts[::-1], return_index=True)
            last = dict(zip(np.unique(ts[::-1]), len(ts) - 1 - first_rev))
            ev = np.where(es)[0]
            if ev.size == 0:
                continue
            pos = np.array([last[ts[i]] for i in ev])
            m = S1[pos] / S0[pos]
            U += np.sum(xs[ev] - m)
            I += np.sum(S2[pos] / S0[pos] - m ** 2)
        if I <= 0:
            return 1.0
        step = U / I
        b += step
        if abs(step) < tol:
            break
    return float(b)


class Recalibrator:
    """mode 'in_large' (slope fixed at 1) or 'slope' (slope estimated)."""

    def __init__(self, mode: str = "slope"):
        self.mode = mode
        self.center_ = 0.0
        self.beta_ = 1.0
        self.times_ = None
        self.H0_ = None

    def fit(self, event, time, pi, horizon=None):
        pi = np.asarray(pi, float)
        self.center_ = float(pi.mean())
        pic = pi - self.center_
        if self.mode == "slope":
            e_h, t_h = _censor(event, time, horizon)
            b, _, _ = cox_fit(e_h, t_h, pic[:, None])
            self.beta_ = float(b[0]) if np.isfinite(b[0]) else 1.0
        else:
            self.beta_ = 1.0
        self.times_, self.H0_ = breslow_baseline(event, time, self.beta_ * pic)
        return self

    def survival_at(self, pi, times_days):
        pic = np.asarray(pi, float) - self.center_
        H0_t = _eval_H0(self.times_, self.H0_, times_days)
        return np.exp(-np.outer(np.exp(self.beta_ * pic), H0_t))


class StratifiedRecalibrator:
    """Limited revision: a separate baseline per stratum, shared slope on the PI.

    Strata with fewer than 20 patients or 5 events use the pooled baseline.
    """

    def __init__(self, min_n=20, min_events=5):
        self.min_n, self.min_events = min_n, min_events
        self.global_ = Recalibrator(mode="slope")
        self.by_stratum_ = {}

    def fit(self, event, time, pi, strata, horizon=None):
        event = np.asarray(event).astype(bool); time = np.asarray(time, float)
        pi = np.asarray(pi, float); strata = np.asarray(strata).astype(str)
        center = float(pi.mean())
        pic = pi - center
        big = np.array([(strata == s).sum() >= self.min_n and event[strata == s].sum() >= self.min_events
                        for s in strata])
        lab = np.where(big, strata, "_pooled")
        e_h, t_h = _censor(event, time, horizon)
        beta = _stratified_slope(e_h, t_h, pic, lab)
        self.global_.center_, self.global_.beta_ = center, beta
        self.global_.times_, self.global_.H0_ = breslow_baseline(event, time, beta * pic)
        for s in np.unique(lab):
            if s == "_pooled":
                continue
            m = lab == s
            r = Recalibrator("in_large")
            r.center_, r.beta_ = center, beta
            r.times_, r.H0_ = breslow_baseline(event[m], time[m], beta * pic[m])
            self.by_stratum_[s] = r
        return self

    def survival_at(self, pi, strata, times_days):
        pi = np.asarray(pi, float); strata = np.asarray(strata).astype(str)
        out = np.empty((len(pi), len(np.atleast_1d(times_days))))
        for s in np.unique(strata):
            m = strata == s
            out[m] = self.by_stratum_.get(s, self.global_).survival_at(pi[m], times_days)
        return out
