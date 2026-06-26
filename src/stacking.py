"""Leakage-controlled stacked survival ensemble (Section 12).

Procedure
---------
1. Repeated K-fold cross-validation within TCGA produces out-of-fold (OOF) risk
   scores for every base learner and every TCGA patient. A base learner never
   predicts a patient that was in its own training fold.
2. The OOF risk-score matrix is the design matrix for a simple penalised-Cox
   meta-learner. Fitting the meta-learner on OOF (not in-sample) predictions is
   what prevents the ensemble's internal performance from being optimistically
   biased.
3. Each base learner is then refitted on the full TCGA cohort.
4. For an external cohort, each refitted base learner produces a risk score;
   these are standardised with the OOF scaling and passed to the frozen
   meta-learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from .preprocessing import make_encoder, survival_y
from .survival_models import SurvivalModel, _new_estimator, _select_coxnet_alpha
from .utils import GLOBAL_SEED, get_logger

log = get_logger("stacking")


def _fit_base_on_index(name, df, feature_set, idx, coxnet_alpha=None):
    """Fit a base learner on a subset of TCGA rows; return a SurvivalModel."""
    sub = df.iloc[idx]
    return SurvivalModel.fit(name, sub, feature_set, coxnet_alpha=coxnet_alpha)


@dataclass
class StackedSurvivalModel:
    name: str = "stacked"
    base_names: tuple = ("cox", "coxnet", "rsf", "gbsa")
    feature_set: str = "B_clinical_molecular"
    base_models_: dict = field(default_factory=dict)
    meta_: object = None
    oof_mean_: np.ndarray = None
    oof_std_: np.ndarray = None

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        feature_set: str,
        base_names=("cox", "coxnet", "rsf", "gbsa"),
        n_splits: int = 5,
        n_repeats: int = 5,
    ) -> "StackedSurvivalModel":
        self = cls(base_names=tuple(base_names), feature_set=feature_set)
        n = len(df)
        event, time = survival_y(df)
        # Select the elastic-net penalty once on the full development cohort and
        # reuse it across folds. This is an in-TCGA hyperparameter choice; the
        # external cohort is not involved.
        cox_alpha = None
        if "coxnet" in base_names:
            from .preprocessing import make_encoder
            from .survival_models import _select_coxnet_alpha
            from sksurv.util import Surv
            enc = make_encoder(feature_set).fit(df)
            Xf = enc.transform(df).to_numpy()
            cox_alpha = _select_coxnet_alpha(Xf, Surv.from_arrays(event, time), event, time)
        oof_sum = {b: np.zeros(n) for b in base_names}
        oof_cnt = np.zeros(n)
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=GLOBAL_SEED)
        for fold, (tr, va) in enumerate(rkf.split(np.arange(n))):
            for b in base_names:
                m = _fit_base_on_index(b, df, feature_set, tr, coxnet_alpha=cox_alpha)
                oof_sum[b][va] += m.risk_score(df.iloc[va])
            oof_cnt[va] += 1
        oof = np.column_stack([oof_sum[b] / np.maximum(oof_cnt, 1) for b in base_names])
        self.oof_mean_ = oof.mean(axis=0)
        self.oof_std_ = oof.std(axis=0)
        self.oof_std_[self.oof_std_ == 0] = 1.0
        Z = (oof - self.oof_mean_) / self.oof_std_
        # Penalised-Cox meta-learner (ridge) over standardised base risk scores.
        self.meta_ = CoxPHSurvivalAnalysis(alpha=1.0)
        self.meta_.fit(Z, Surv.from_arrays(event, time))
        # Refit base learners on full TCGA (reusing the selected penalty).
        self.base_models_ = {
            b: SurvivalModel.fit(b, df, feature_set, coxnet_alpha=cox_alpha) for b in base_names
        }
        log.info("stacked ensemble fitted: bases=%s", ",".join(base_names))
        return self

    def _meta_design(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.column_stack([self.base_models_[b].risk_score(df) for b in self.base_names])
        return (scores - self.oof_mean_) / self.oof_std_

    def risk_score(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_.predict(self._meta_design(df))).ravel()

    def survival_at(self, df: pd.DataFrame, times_days) -> np.ndarray:
        Z = self._meta_design(df)
        surv_fns = self.meta_.predict_survival_function(Z)
        times = np.atleast_1d(times_days).astype(float)
        out = np.empty((len(surv_fns), len(times)))
        for i, fn in enumerate(surv_fns):
            out[i] = fn(np.clip(times, fn.x.min(), fn.x.max()))
        return out
