"""Tests for survival metric primitives."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.validation import _cox_univariate_slope, calibration_in_the_large, harrell_c, km_survival_at  # noqa: E402


def test_harrell_c_perfect_and_random():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.uniform(1, 100, n)
    event = np.ones(n, bool)
    # risk perfectly inversely related to survival time -> C = 1
    risk = -time
    assert harrell_c(event, time, risk) > 0.99
    # random risk -> C near 0.5
    assert abs(harrell_c(event, time, rng.normal(size=n)) - 0.5) < 0.1


def test_fast_cox_slope_matches_lifelines():
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    # generate exponential times with hazard exp(0.7 x)
    time = rng.exponential(1 / np.exp(0.7 * x))
    event = (rng.uniform(size=n) < 0.8).astype(int)
    fast = _cox_univariate_slope(event, time, x)
    d = pd.DataFrame({"t": time, "e": event, "x": x})
    ll = CoxPHFitter().fit(d, "t", "e").params_["x"]
    assert abs(fast - ll) < 0.03


def test_km_and_citl_bounds():
    rng = np.random.default_rng(2)
    n = 300
    time = rng.uniform(1, 1000, n)
    event = (rng.uniform(size=n) < 0.5).astype(int)
    s = km_survival_at(event, time, 365)
    assert 0.0 <= s <= 1.0
    pred = np.full(n, 0.6)
    citl = calibration_in_the_large(event, time, pred, 365)
    assert -1.0 <= citl <= 1.0
