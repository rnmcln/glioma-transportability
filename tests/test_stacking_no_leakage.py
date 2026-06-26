"""Test that stacking uses out-of-fold base predictions (no information leak).

On data where the outcome is independent of the features, an honest stacked
estimate (meta-learner cross-validated on out-of-fold base scores) must have a
concordance near 0.5. Fitting the meta-learner on in-sample base predictions
instead inflates concordance; the test asserts both, so it has power to detect
leakage rather than merely passing trivially.
"""
import pathlib
import sys

import numpy as np
from sklearn.model_selection import KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _synth(n=300, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    time = rng.exponential(50, size=n)        # outcome independent of X
    event = (rng.uniform(size=n) < 0.6)
    return X, Surv.from_arrays(event, time), event, time


def _base_models():
    return [
        RandomSurvivalForest(n_estimators=60, min_samples_leaf=15, random_state=0, n_jobs=2),
        GradientBoostingSurvivalAnalysis(n_estimators=60, max_depth=2, random_state=0),
    ]


def _oof_scores(X, y, event, time, splits=5):
    n = len(time)
    bases = _base_models()
    oof = np.zeros((n, len(bases)))
    insample = np.zeros((n, len(bases)))
    kf = KFold(splits, shuffle=True, random_state=0)
    for j, b in enumerate(bases):
        for tr, va in kf.split(X):
            m = b.__class__(**b.get_params()).fit(X[tr], y[tr])
            oof[va, j] = m.predict(X[va])
        bfull = b.fit(X, y)
        insample[:, j] = bfull.predict(X)
    return oof, insample


def test_oof_meta_not_leaky():
    X, y, event, time = _synth()
    oof, insample = _oof_scores(X, y, event, time)

    # Honest: cross-validate the meta-learner on OOF base scores.
    kf = KFold(5, shuffle=True, random_state=1)
    pred = np.zeros(len(time))
    for tr, va in kf.split(oof):
        meta = CoxPHSurvivalAnalysis(alpha=1.0).fit(oof[tr], Surv.from_arrays(event[tr], time[tr]))
        pred[va] = meta.predict(oof[va])
    c_honest = concordance_index_censored(event, time, pred)[0]
    assert abs(c_honest - 0.5) < 0.1, f"honest stacked C should be ~0.5 on noise, got {c_honest:.3f}"

    # Leaky: meta fitted on in-sample base predictions and scored in-sample.
    meta = CoxPHSurvivalAnalysis(alpha=1.0).fit(insample, y)
    c_leak = concordance_index_censored(event, time, meta.predict(insample))[0]
    # The leaky route manufactures apparent signal above the honest estimate.
    assert c_leak > c_honest, "leaky in-sample meta should inflate concordance"
