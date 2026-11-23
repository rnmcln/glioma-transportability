"""Model-updating ladder, evaluated by cross-validation within a CGGA release.

Stages
------
0  Frozen TCGA model applied directly to CGGA.
1  Recalibration-in-the-large: re-estimate the baseline survival in CGGA with the
   prognostic-index slope fixed at 1 (corrects the average level).
2  Logistic-style slope update: re-estimate the baseline survival and a slope on
   the prognostic index (corrects level and spread).
3  Limited revision: a separate recalibrated baseline per integrated-class
   stratum with a shared slope (model feature weights unchanged).
4  Full refit of the model class in CGGA (comparator; no longer external
   validation of the TCGA model).

Stages 1-3 operate on the prognostic index PI = log(-log S_hat(24 months | x))
of the frozen model (see calibration.py). Stages 1-4 are estimated on training folds and evaluated on held-out folds
(5-fold cross-validation), so reported values are not in-sample. All stages are
summarised on the same scale: Harrell C of predicted 24-month risk,
calibration-in-the-large, calibration slope and ICI at 24 months, and the
integrated Brier score over 1-36 months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .calibration import Recalibrator, StratifiedRecalibrator, prognostic_index
from .utils import GLOBAL_SEED, GRID_DAYS, PRIMARY_GRID_IDX, PRIMARY_HORIZON_DAYS, get_logger
from .validation import (
    calibration_slope_at, harrell_c, ici_at, integrated_brier, km_survival_at,
)

log = get_logger("updating")
STAGES = ["0_frozen", "1_in_large", "2_slope", "3_revision", "4_refit"]


def _stage_metrics(event, time, surv_grid):
    sp = surv_grid[:, PRIMARY_GRID_IDX]
    obs = km_survival_at(event, time, PRIMARY_HORIZON_DAYS)
    ic = ici_at(event, time, sp, PRIMARY_HORIZON_DAYS)
    return dict(
        harrell_c=harrell_c(event, time, 1.0 - sp),
        obs_24m=obs,
        pred_mean_24m=float(np.mean(sp)),
        citl_24m=obs - float(np.mean(sp)),
        slope_24m=calibration_slope_at(event, time, sp, PRIMARY_HORIZON_DAYS),
        ici_24m=ic["ici"],
        ibs=integrated_brier(event, time, surv_grid),
    )


def run_ladder(df, frozen_model, refit_fn=None, n_splits=5, do_stage4=True):
    """Evaluate Stages 0-4 for one frozen model on one CGGA cohort."""
    df = df.reset_index(drop=True)
    event = df.os_event.to_numpy().astype(bool)
    time = df.os_time_days.to_numpy().astype(float)
    strata = df.integrated_class_harmonised.to_numpy()
    grid = list(GRID_DAYS)
    n, G = len(df), len(grid)
    frozen = frozen_model.survival_at(df, grid)
    pi = prognostic_index(frozen[:, PRIMARY_GRID_IDX])

    preds = {"0_frozen": frozen}
    cv = {s: np.zeros((n, G)) for s in STAGES[1:4]}
    if do_stage4 and refit_fn is not None:
        cv["4_refit"] = np.zeros((n, G))
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED).split(np.arange(n)):
        cv["1_in_large"][va] = Recalibrator("in_large").fit(event[tr], time[tr], pi[tr]).survival_at(pi[va], grid)
        cv["2_slope"][va] = Recalibrator("slope").fit(event[tr], time[tr], pi[tr], PRIMARY_HORIZON_DAYS).survival_at(pi[va], grid)
        cv["3_revision"][va] = StratifiedRecalibrator().fit(
            event[tr], time[tr], pi[tr], strata[tr], PRIMARY_HORIZON_DAYS).survival_at(pi[va], strata[va], grid)
        if "4_refit" in cv:
            cv["4_refit"][va] = refit_fn(df.iloc[tr]).survival_at(df.iloc[va], grid)
    preds.update(cv)
    rows = [dict(stage=s, **_stage_metrics(event, time, preds[s])) for s in STAGES if s in preds]
    return pd.DataFrame(rows), preds
