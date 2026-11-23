#!/usr/bin/env python
"""07_internal_validation.py - cross-validated internal validation in TCGA.

Repeated 5 x 5 cross-validation produces out-of-fold (OOF) risk scores and OOF
survival predictions (1-36 month grid) for every base learner and reference
model; each patient's OOF prediction is the mean over the five repeats. The
stacked ensemble's OOF prediction is obtained by cross-validating the
meta-learner on the OOF base scores. Metrics are computed on the pooled OOF
predictions with patient-level bootstrap intervals. Internal validation is the
benchmark against which external performance is compared.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.preprocessing import survival_y  # noqa: E402
from src.registry import CODE_FS, comparison_pairs  # noqa: E402
from src.survival_models import MODEL_NAMES, SurvivalModel, predict_all  # noqa: E402
from src.utils import GLOBAL_SEED, GRID_DAYS, HORIZON_GRID_IDX, get_logger, get_paths, set_global_seed  # noqa: E402
from src.validation import bootstrap_panel  # noqa: E402

log = get_logger("07_internal")
N_SPLITS, N_REPEATS, N_BOOT = 5, 5, 500
JOBS = [("B", m) for m in MODEL_NAMES] + [("A", m) for m in MODEL_NAMES] + [("R", "cox"), ("RA", "cox"), ("RAG", "cox")]


def main() -> int:
    set_global_seed()
    paths = get_paths()
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet").reset_index(drop=True)
    n, G = len(tcga), len(GRID_DAYS)
    event, time_d = survival_y(tcga)
    acc = {f"{c}:{m}": dict(risk=np.zeros(n), surv=np.zeros((n, G)), cnt=np.zeros(n)) for c, m in JOBS}
    folds = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                               random_state=GLOBAL_SEED).split(np.arange(n)))
    t0 = time.time()
    for fid, (tr, va) in enumerate(folds):
        for code, name in JOBS:
            m = SurvivalModel.fit(name, tcga.iloc[tr], CODE_FS[code])
            r, s = predict_all(m, tcga.iloc[va])
            a = acc[f"{code}:{name}"]
            a["risk"][va] += r; a["surv"][va] += s; a["cnt"][va] += 1
        log.info("fold %d/%d (%.0fs)", fid + 1, len(folds), time.time() - t0)
    oof = {k: (v["risk"] / v["cnt"], v["surv"] / v["cnt"][:, None]) for k, v in acc.items()}

    # Stacked OOF: cross-validate the meta-learner on OOF base risk scores.
    for code in ["B", "A"]:
        Z = np.column_stack([oof[f"{code}:{m}"][0] for m in MODEL_NAMES])
        mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1
        Zs = (Z - mu) / sd
        s_risk, s_surv = np.zeros(n), np.zeros((n, G))
        for tr, va in KFold(5, shuffle=True, random_state=GLOBAL_SEED).split(Zs):
            meta = CoxPHSurvivalAnalysis(alpha=1.0).fit(Zs[tr], Surv.from_arrays(event[tr], time_d[tr]))
            s_risk[va] = meta.predict(Zs[va])
            for k, fn in zip(va, meta.predict_survival_function(Zs[va])):
                s_surv[k] = fn(np.clip(np.array(GRID_DAYS), fn.x.min(), fn.x.max()))
        oof[f"{code}:stacked"] = (s_risk, s_surv)

    np.savez_compressed(paths.data_processed / "oof_tcga.npz",
                        **{f"{k}|risk": v[0] for k, v in oof.items()},
                        **{f"{k}|surv": v[1] for k, v in oof.items()},
                        patient_id=tcga.patient_id.to_numpy())
    preds = {k: (v[0], v[1][:, list(HORIZON_GRID_IDX)], v[1]) for k, v in oof.items()}
    summary, diffs = bootstrap_panel(event, time_d, preds, n_boot=N_BOOT, pairs=comparison_pairs())
    summary.insert(0, "cohort", "TCGA_internal")
    diffs.insert(0, "cohort", "TCGA_internal")
    summary.to_csv(paths.tables / "internal_validation_metrics.csv", index=False)
    diffs.to_csv(paths.tables / "internal_validation_differences.csv", index=False)
    v = summary[summary.metric.isin(["harrell_c", "slope_24m", "citl_24m", "ici_24m", "ibs"])]
    log.info("internal validation complete\n%s",
             v.pivot(index="model", columns="metric", values="point").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
