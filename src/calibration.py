"""Recalibration primitives based on a frozen prognostic index (PI).

Each fitted model exposes a risk score that is treated as a prognostic index.
Recalibration re-estimates, in the target cohort, a Breslow baseline survival
function and (optionally) a slope on the PI, while leaving the original model's
feature weights untouched. This is the van Houwelingen recalibration framework
and underpins Stages 1-3 of the updating ladder.
"""
from __future__ import annotations

import numpy as np

from .validation import _cox_univariate_slope


def breslow_baseline(event, time, pi_centered, beta):
    """Breslow estimate of the cumulative baseline hazard given offset beta*pi.

    Returns (unique_times, H0) where S0(t) = exp(-H0). The baseline corresponds
    to a subject at the mean PI (pi_centered has mean 0).
    """
    t = np.asarray(time, float)
    e = np.asarray(event).astype(bool)
    pic = np.asarray(pi_centered, float)
    w = np.exp(beta * pic)
    order = np.argsort(t)
    t_s, e_s, w_s = t[order], e[order], w[order]
    # risk set sum of weights for each subject = sum of w for time >= t
    rev_cum_w = np.cumsum(w_s[::-1])[::-1]
    uniq = np.unique(t_s[e_s])
    H0 = np.zeros(len(uniq))
    h = 0.0
    for k, tt in enumerate(uniq):
        d = int(np.sum(e_s & (t_s == tt)))
        denom = rev_cum_w[np.searchsorted(t_s, tt, side="left")]
        if denom > 0:
            h += d / denom
        H0[k] = h
    return uniq, H0


class Recalibrator:
    """Fit and apply a recalibrated survival predictor on a PI.

    mode:
        'in_large' -> slope fixed at 1, baseline re-estimated (corrects level);
        'slope'    -> slope estimated and baseline re-estimated (level + spread).
    """

    def __init__(self, mode: str = "slope"):
        self.mode = mode
        self.pi_mean_ = 0.0
        self.pi_std_ = 1.0
        self.beta_ = 1.0
        self.times_ = None
        self.H0_ = None

    def fit(self, event, time, pi):
        # Standardise the prognostic index so that exp(beta*pi) is numerically
        # stable for risk scores not on the log-hazard scale (e.g. forests).
        pi = np.asarray(pi, float)
        self.pi_mean_ = float(pi.mean())
        self.pi_std_ = float(pi.std()) or 1.0
        pic = (pi - self.pi_mean_) / self.pi_std_
        if self.mode == "slope":
            self.beta_ = _cox_univariate_slope(event, time, pic)
            if not np.isfinite(self.beta_):
                self.beta_ = 1.0
        else:
            self.beta_ = 1.0
        self.times_, self.H0_ = breslow_baseline(event, time, pic, self.beta_)
        return self

    def survival_at(self, pi, times_days):
        pic = (np.asarray(pi, float) - self.pi_mean_) / self.pi_std_
        times = np.atleast_1d(times_days).astype(float)
        idx = np.searchsorted(self.times_, times, side="right") - 1
        H0_t = np.where(idx >= 0, self.H0_[np.clip(idx, 0, len(self.H0_) - 1)], 0.0)
        lp = np.exp(self.beta_ * pic)
        # S(t|x) = exp(-H0(t) * exp(beta*pic))
        return np.exp(-np.outer(lp, H0_t))


class StratifiedRecalibrator:
    """Stage 3 limited revision: a separate recalibrated baseline per stratum
    (with a shared estimated slope), allowing the baseline survival to differ by
    major subgroup while keeping the model's feature weights fixed.
    """

    def __init__(self):
        self.global_ = Recalibrator(mode="slope")
        self.by_stratum_ = {}

    def fit(self, event, time, pi, strata):
        event = np.asarray(event); time = np.asarray(time); pi = np.asarray(pi, float)
        strata = np.asarray(strata)
        self.global_.fit(event, time, pi)
        beta = self.global_.beta_
        pim, pis = self.global_.pi_mean_, self.global_.pi_std_
        for s in np.unique(strata):
            m = strata == s
            if m.sum() >= 20 and np.asarray(event)[m].sum() >= 5:
                r = Recalibrator(mode="in_large")
                r.beta_ = beta
                r.pi_mean_ = pim
                r.pi_std_ = pis
                pic = (pi[m] - pim) / pis
                r.times_, r.H0_ = breslow_baseline(event[m], time[m], pic, beta)
                self.by_stratum_[s] = r
        return self

    def survival_at(self, pi, strata, times_days):
        pi = np.asarray(pi, float); strata = np.asarray(strata)
        times = np.atleast_1d(times_days).astype(float)
        out = np.empty((len(pi), len(times)))
        for i in range(len(pi)):
            r = self.by_stratum_.get(strata[i], self.global_)
            out[i] = r.survival_at(np.array([pi[i]]), times)[0]
        return out
