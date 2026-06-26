"""Survival model classes trained exclusively in TCGA.

All hyperparameters are fixed a priori or selected by internal cross-validation
within TCGA. CGGA is never used for fitting, tuning, or model selection.

Each model is wrapped in :class:`SurvivalModel`, which couples a fitted design
encoder with a fitted survival estimator and exposes:

* ``risk_score(df)``    -- a scalar risk score (higher = worse prognosis);
* ``survival_at(df, t)`` -- predicted S(t | x) at requested day horizons.

The risk score and the survival function are the two quantities required for,
respectively, discrimination metrics and calibration/Brier metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from .preprocessing import DesignEncoder, make_encoder, survival_y
from .utils import GLOBAL_SEED, get_logger

log = get_logger("survival_models")

MODEL_NAMES = ["cox", "coxnet", "rsf", "gbsa"]
MODEL_LABELS = {
    "cox": "Cox PH",
    "coxnet": "Penalised Cox (elastic net)",
    "rsf": "Random survival forest",
    "gbsa": "Gradient-boosted survival",
    "stacked": "Stacked ensemble",
}


def _new_estimator(name: str):
    if name == "cox":
        # Small ridge penalty for numerical stability with one-hot predictors.
        return CoxPHSurvivalAnalysis(alpha=1.0)
    if name == "coxnet":
        return CoxnetSurvivalAnalysis(
            l1_ratio=0.9, fit_baseline_model=True, alpha_min_ratio=0.01, max_iter=100000
        )
    if name == "rsf":
        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=4,
            random_state=GLOBAL_SEED,
        )
    if name == "gbsa":
        return GradientBoostingSurvivalAnalysis(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=GLOBAL_SEED,
        )
    raise ValueError(f"unknown model {name}")


def _select_coxnet_alpha(X: np.ndarray, y, event, time, n_splits: int = 5) -> float:
    """Choose the elastic-net penalty by internal K-fold concordance (TCGA only)."""
    from sklearn.model_selection import KFold

    base = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100000)
    base.fit(X, y)
    alphas = base.alphas_
    # Evaluate a coarse subset of the path for speed.
    cand = alphas[:: max(1, len(alphas) // 12)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    best_a, best_c = cand[0], -np.inf
    for a in cand:
        cs = []
        for tr, va in kf.split(X):
            try:
                m = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[a], max_iter=100000)
                m.fit(X[tr], Surv.from_arrays(event[tr], time[tr]))
                r = m.predict(X[va])
                cs.append(concordance_index_censored(event[va], time[va], r)[0])
            except Exception:
                cs.append(np.nan)
        mc = np.nanmean(cs)
        if mc > best_c:
            best_c, best_a = mc, a
    log.info("coxnet alpha selected=%.5g (internal CV C=%.3f)", best_a, best_c)
    return float(best_a)


@dataclass
class SurvivalModel:
    name: str
    encoder: DesignEncoder
    estimator: object
    train_times_: np.ndarray = None  # event times available for survival fn

    @classmethod
    def fit(cls, name: str, df: pd.DataFrame, feature_set: str,
            coxnet_alpha: float | None = None) -> "SurvivalModel":
        """Fit a base learner on TCGA.

        ``coxnet_alpha`` may be supplied to bypass internal penalty selection
        (used during stacking out-of-fold fitting, where the penalty is fixed at
        the value selected once on the full development cohort).
        """
        enc = make_encoder(feature_set).fit(df)
        X = enc.transform(df).to_numpy()
        event, time = survival_y(df)
        y = Surv.from_arrays(event, time)
        if name == "coxnet":
            alpha = coxnet_alpha if coxnet_alpha is not None else _select_coxnet_alpha(X, y, event, time)
            est = CoxnetSurvivalAnalysis(
                l1_ratio=0.9, alphas=[alpha], fit_baseline_model=True, max_iter=100000
            )
        else:
            est = _new_estimator(name)
        est.fit(X, y)
        return cls(name=name, encoder=enc, estimator=est, train_times_=np.unique(time[event]))

    def risk_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.encoder.transform(df).to_numpy()
        return np.asarray(self.estimator.predict(X)).ravel()

    def survival_at(self, df: pd.DataFrame, times_days) -> np.ndarray:
        """Return an (n_samples, n_times) matrix of predicted S(t | x)."""
        X = self.encoder.transform(df).to_numpy()
        surv_fns = self.estimator.predict_survival_function(X)
        times = np.atleast_1d(times_days).astype(float)
        out = np.empty((len(surv_fns), len(times)))
        for i, fn in enumerate(surv_fns):
            out[i] = fn(np.clip(times, fn.x.min(), fn.x.max()))
        return out
