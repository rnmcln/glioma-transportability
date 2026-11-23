"""Tests for survival metric primitives."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.validation import (calibration_in_the_large, calibration_slope_at, cox_fit,  # noqa: E402
                            harrell_c, harrell_c_pairs, ici_at, km_survival_at)


def _sim(n=2000, beta=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    lam = 0.002
    t_ev = rng.exponential(1 / (lam * np.exp(beta * x)))
    t_c = rng.uniform(200, 3000, n)
    time = np.minimum(t_ev, t_c)
    event = (t_ev <= t_c).astype(int)
    return x, time, event, lam


def test_harrell_c_perfect_and_random():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.uniform(1, 100, n)
    event = np.ones(n, bool)
    assert harrell_c(event, time, -time) > 0.99
    assert abs(harrell_c(event, time, rng.normal(size=n)) - 0.5) < 0.1


def test_cox_fit_matches_lifelines():
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 2))
    time = rng.exponential(1 / np.exp(0.7 * X[:, 0] - 0.4 * X[:, 1]))
    event = (rng.uniform(size=n) < 0.8).astype(int)
    b, _, _ = cox_fit(event, time, X)
    d = pd.DataFrame({"t": time, "e": event, "x0": X[:, 0], "x1": X[:, 1]})
    ll = CoxPHFitter().fit(d, "t", "e").params_.to_numpy()
    assert np.allclose(b, ll, atol=0.02)


def test_calibration_slope_and_ici_for_true_model():
    x, time, event, lam = _sim()
    t = 365.0
    s_true = np.exp(-lam * t * np.exp(0.8 * x))
    assert abs(calibration_slope_at(event, time, s_true, t) - 1.0) < 0.12
    assert ici_at(event, time, s_true, t)["ici"] < 0.02
    # predictions that are too extreme (doubled linear predictor) give slope < 1
    s_ext = np.exp(-lam * t * np.exp(1.6 * x))
    assert calibration_slope_at(event, time, s_ext, t) < 0.7


def test_pair_decomposition_consistent():
    x, time, event, _ = _sim(n=600, seed=3)
    g = np.where(x > 0, "a", "b")
    d = harrell_c_pairs(event, time, x, g)
    assert abs(d["c_overall"] - harrell_c(event, time, x)) < 0.005
    assert 0 < d["share_between"] < 1


def test_km_and_citl_bounds():
    rng = np.random.default_rng(2)
    n = 300
    time = rng.uniform(1, 1000, n)
    event = (rng.uniform(size=n) < 0.5).astype(int)
    assert 0.0 <= km_survival_at(event, time, 365) <= 1.0
    assert -1.0 <= calibration_in_the_large(event, time, np.full(n, 0.6), 365) <= 1.0


def test_fast_context_matches_scikit_survival():
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                cumulative_dynamic_auc, integrated_brier_score)
    from sksurv.util import Surv
    from src.validation import EvalContext
    x, time, event, lam = _sim(n=500, seed=5)
    time = np.round(time)  # induce ties
    e = event.astype(bool)
    y = Surv.from_arrays(e, time)
    ctx = EvalContext(e, time)
    assert abs(ctx.harrell(x) - concordance_index_censored(e, time, x)[0]) < 1e-10
    tau = np.quantile(time[e], 0.9)
    assert abs(ctx.uno(x) - concordance_index_ipcw(y, y, x, tau=tau)[0]) < 1e-10
    assert abs(ctx.auc(x, 365.0) - cumulative_dynamic_auc(y, y, x, [365.0])[0][0]) < 1e-10
    S = np.exp(-lam * np.outer(np.exp(0.8 * x), ctx.grid))
    k = ctx.grid_keep
    assert abs(ctx.ibs(S) - integrated_brier_score(y, y, S[:, k], ctx.grid[k])) < 1e-10
