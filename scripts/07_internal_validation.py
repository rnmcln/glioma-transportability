#!/usr/bin/env python
"""07_internal_validation.py - cross-validated internal validation in TCGA.

Internal validation is secondary to external validation (Section 13); it exists
to quantify the internal-to-external performance change. A single repeated 5x5
cross-validation produces out-of-fold (OOF) risk scores and OOF survival
predictions for the four base learners. The stacked ensemble's OOF prediction is
obtained by cross-validating the penalised-Cox meta-learner on the OOF base
scores, which avoids nested refitting while keeping the estimate honest.

The script is checkpointed: it processes folds until a wall-time budget is
reached, persists state, and resumes on re-invocation. Run repeatedly until it
reports 'internal CV complete'.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.survival_models import MODEL_NAMES, SurvivalModel  # noqa: E402
from src.preprocessing import survival_y  # noqa: E402
from src.utils import GLOBAL_SEED, HORIZONS_DAYS, get_logger, get_paths, set_global_seed  # noqa: E402
from src.validation import (  # noqa: E402
    calibration_in_the_large, calibration_slope, harrell_c, integrated_brier, km_survival_at,
    time_auc, uno_c,
)

log = get_logger("07_internal")
N_SPLITS, N_REPEATS = 5, 5
STATE = "internal_cv_state_{fs}.pkl"


def _empty_state(n):
    H = len(HORIZONS_DAYS)
    base = {m: dict(risk=np.zeros(n), surv=np.zeros((n, H)), cnt=np.zeros(n)) for m in MODEL_NAMES}
    return dict(done_folds=[], base=base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--feature-set", default="B_clinical_molecular")
    args = ap.parse_args()
    set_global_seed()
    paths = get_paths()
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    n = len(tcga)
    event, time_d = survival_y(tcga)

    spath = paths.data_processed / STATE.format(fs=args.feature_set)
    state = pickle.load(open(spath, "rb")) if spath.exists() else _empty_state(n)

    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=GLOBAL_SEED).split(np.arange(n)))
    t0 = time.time()
    for fid, (tr, va) in enumerate(folds):
        if fid in state["done_folds"]:
            continue
        if time.time() - t0 > args.budget:
            break
        for name in MODEL_NAMES:
            m = SurvivalModel.fit(name, tcga.iloc[tr], args.feature_set)
            state["base"][name]["risk"][va] += m.risk_score(tcga.iloc[va])
            state["base"][name]["surv"][va] += m.survival_at(tcga.iloc[va], list(HORIZONS_DAYS))
            state["base"][name]["cnt"][va] += 1
        state["done_folds"].append(fid)
        pickle.dump(state, open(spath, "wb"))
        log.info("fold %d/%d done (%.1fs)", fid + 1, len(folds), time.time() - t0)

    if len(state["done_folds"]) < len(folds):
        log.info("checkpoint: %d/%d folds; re-run to continue", len(state["done_folds"]), len(folds))
        return 0

    # All folds done -> aggregate OOF and compute metrics.
    log.info("internal CV complete; computing metrics")
    rows, oof_risk = [], {}
    for name in MODEL_NAMES:
        cnt = np.maximum(state["base"][name]["cnt"], 1)
        risk = state["base"][name]["risk"] / cnt
        surv = state["base"][name]["surv"] / cnt[:, None]
        oof_risk[name] = risk
        rows.append(_metric_row(name, event, time_d, risk, surv))

    # Stacked OOF via meta-learner CV on OOF base scores.
    Z = np.column_stack([oof_risk[m] for m in MODEL_NAMES])
    mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1
    Zs = (Z - mu) / sd
    s_risk = np.zeros(n); s_surv = np.zeros((n, len(HORIZONS_DAYS)))
    for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED).split(Zs):
        meta = CoxPHSurvivalAnalysis(alpha=1.0).fit(Zs[tr], Surv.from_arrays(event[tr], time_d[tr]))
        s_risk[va] = meta.predict(Zs[va])
        fns = meta.predict_survival_function(Zs[va])
        for k, fn in zip(va, fns):
            s_surv[k] = fn(np.clip(np.array(HORIZONS_DAYS, float), fn.x.min(), fn.x.max()))
    rows.append(_metric_row("stacked", event, time_d, s_risk, s_surv))

    out = pd.DataFrame(rows)
    out.insert(0, "cohort", "TCGA_internal_cv")
    out.to_csv(paths.tables / f"table4_internal_validation_{args.feature_set}.csv", index=False)
    log.info("wrote table4 (%s)\n%s", args.feature_set, out.round(3).to_string(index=False))
    return 0


def _metric_row(name, event, time_d, risk, surv):
    row = dict(model=name)
    row["harrell_c"] = harrell_c(event, time_d, risk)
    row["uno_c"] = uno_c(event, time_d, risk)
    row["ibs"] = integrated_brier(event, time_d, surv, list(HORIZONS_DAYS))
    row["calibration_slope"] = calibration_slope(event, time_d, risk)
    auc = time_auc(event, time_d, risk, list(HORIZONS_DAYS))
    for j, t in enumerate(HORIZONS_DAYS):
        mo = int(round(t / 30.4375))
        row[f"auc_{mo}m"] = auc.get(int(t), float("nan"))
        row[f"citl_{mo}m"] = calibration_in_the_large(event, time_d, surv[:, j], t)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
