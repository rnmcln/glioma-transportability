"""Model-updating ladder (Section 15), evaluated by cross-validation in CGGA.

Stages
------
0  Frozen TCGA model applied directly to CGGA.
1  Recalibration-in-the-large: re-estimate the baseline survival in CGGA with the
   prognostic-index slope fixed at 1 (corrects the average level).
2  Slope update: re-estimate the baseline survival and a slope on the PI.
3  Limited revision: a separate recalibrated baseline per integrated-class
   stratum with a shared slope (feature weights still fixed).
4  Full refit of the model class in CGGA (upper-bound comparator).

Recalibration parameters (Stages 1-3) and the full refit (Stage 4) are estimated
on training folds and evaluated on held-out folds, so reported gains are not
in-sample. Discrimination, calibration-in-the-large, a calibration slope on the
predicted cumulative hazard, and the integrated Brier score are reported at the
primary horizon and across horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .calibration import Recalibrator, StratifiedRecalibrator
from .utils import GLOBAL_SEED, HORIZONS_DAYS, PRIMARY_HORIZON_DAYS, get_logger
from .validation import (
    _cox_univariate_slope, harrell_c, integrated_brier, km_survival_at,
)

log = get_logger("updating")
_EPS = 1e-6


def _cloglog_lp(surv_at_primary):
    s = np.clip(np.asarray(surv_at_primary, float), _EPS, 1 - _EPS)
    return np.log(-np.log(s))  # complementary log-log of S(t) = log cumulative hazard


def _stage_metrics(event, time, surv_matrix, horizons, primary_idx):
    sp = surv_matrix[:, primary_idx]
    risk = 1.0 - sp
    lp = _cloglog_lp(sp)
    lp = (lp - lp.mean()) / (lp.std() or 1.0)
    return dict(
        harrell_c=harrell_c(event, time, risk),
        citl_primary=km_survival_at(event, time, PRIMARY_HORIZON_DAYS) - float(np.mean(sp)),
        cal_slope=_cox_univariate_slope(event, time, lp),
        ibs=integrated_brier(event, time, surv_matrix, list(horizons)),
        pred_mean_primary=float(np.mean(sp)),
        obs_primary=km_survival_at(event, time, PRIMARY_HORIZON_DAYS),
    )


def run_ladder(df, frozen_model, refit_fn=None, feature_set="B_clinical_molecular",
               horizons=HORIZONS_DAYS, n_splits=5, do_stage4=True) -> pd.DataFrame:
    """Evaluate Stages 0-4 for one model on one CGGA cohort."""
    df = df.reset_index(drop=True)
    event = df.os_event.to_numpy().astype(bool)
    time = df.os_time_days.to_numpy().astype(float)
    strata = df.integrated_class_harmonised.to_numpy()
    pi = frozen_model.risk_score(df)
    primary_idx = list(horizons).index(PRIMARY_HORIZON_DAYS) if PRIMARY_HORIZON_DAYS in horizons else 1
    n = len(df)

    # Stage 0: frozen predictions on the full cohort.
    surv0 = frozen_model.survival_at(df, list(horizons))
    rows = [dict(stage="0_frozen", **_stage_metrics(event, time, surv0, horizons, primary_idx))]

    # Allocate held-out prediction matrices for CV stages.
    H = len(horizons)
    pred = {s: np.zeros((n, H)) for s in ["1_in_large", "2_slope", "3_revision"]}
    if do_stage4 and refit_fn is not None:
        pred["4_refit"] = np.zeros((n, H))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    for tr, va in kf.split(np.arange(n)):
        r1 = Recalibrator("in_large").fit(event[tr], time[tr], pi[tr])
        pred["1_in_large"][va] = r1.survival_at(pi[va], list(horizons))
        r2 = Recalibrator("slope").fit(event[tr], time[tr], pi[tr])
        pred["2_slope"][va] = r2.survival_at(pi[va], list(horizons))
        r3 = StratifiedRecalibrator().fit(event[tr], time[tr], pi[tr], strata[tr])
        pred["3_revision"][va] = r3.survival_at(pi[va], strata[va], list(horizons))
        if "4_refit" in pred:
            rm = refit_fn(df.iloc[tr])
            pred["4_refit"][va] = rm.survival_at(df.iloc[va], list(horizons))

    for s in ["1_in_large", "2_slope", "3_revision"] + (["4_refit"] if "4_refit" in pred else []):
        rows.append(dict(stage=s, **_stage_metrics(event, time, pred[s], horizons, primary_idx)))
    out = pd.DataFrame(rows)
    return out
