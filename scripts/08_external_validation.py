#!/usr/bin/env python
"""08_external_validation.py - frozen external validation on a CGGA release.

For each frozen TCGA-trained model, per-patient risk and survival predictions
are computed once on the CGGA cohort, then a patient-clustered bootstrap gives
confidence intervals for the metric panel. This is the primary analysis
(Section 14): the models are never refitted or tuned on CGGA.

Checkpointed per model. Run repeatedly until 'external validation complete'.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.survival_models import MODEL_NAMES  # noqa: E402
from src.utils import HORIZONS_DAYS, get_logger, get_paths, set_global_seed  # noqa: E402
from src.validation import bootstrap_arrays, evaluate_arrays  # noqa: E402

log = get_logger("08_external")
ALL_MODELS = MODEL_NAMES + ["stacked"]


def _load_cohort(paths, release):
    key = release.lower().replace("_", "")
    df = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return df[df.included_primary_analysis].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="CGGA release key, e.g. mRNAseq_693")
    ap.add_argument("--feature-set", default="B_clinical_molecular")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()
    set_global_seed()
    paths = get_paths()

    df = _load_cohort(paths, args.cohort)
    event = df.os_event.to_numpy().astype(bool)
    time_d = df.os_time_days.to_numpy().astype(float)
    clusters = df.patient_id.to_numpy()
    fs = args.feature_set
    mdir = paths.models / fs

    out_csv = paths.tables / f"table5_external_{args.cohort}_{fs}.csv"
    partial = paths.data_processed / f"ext_{args.cohort}_{fs}_partial.pkl"
    done = pickle.load(open(partial, "rb")) if partial.exists() else {}

    t0 = time.time()
    for name in ALL_MODELS:
        if name in done:
            continue
        if time.time() - t0 > args.budget:
            break
        model = pickle.load(open(mdir / f"{name}.pkl", "rb"))
        risk = model.risk_score(df)
        surv = model.survival_at(df, list(HORIZONS_DAYS))
        boot = bootstrap_arrays(event, time_d, risk, surv, clusters, n_boot=args.n_boot)
        row = {"model": name, "n": len(df), "n_events": int(event.sum())}
        for _, r in boot.iterrows():
            row[r.metric] = r.point
            row[f"{r.metric}_lo"] = r.lo
            row[f"{r.metric}_hi"] = r.hi
        done[name] = row
        pickle.dump(done, open(partial, "wb"))
        log.info("%s done (%.1fs): Harrell C=%.3f [%.3f,%.3f]  CITL24=%.3f",
                 name, time.time() - t0, row["harrell_c"], row["harrell_c_lo"],
                 row["harrell_c_hi"], row.get("citl_24m", float("nan")))

    if len(done) < len(ALL_MODELS):
        log.info("checkpoint: %d/%d models; re-run to continue", len(done), len(ALL_MODELS))
        return 0

    out = pd.DataFrame([done[m] for m in ALL_MODELS])
    out.insert(0, "cohort", args.cohort)
    out.to_csv(out_csv, index=False)
    log.info("external validation complete -> %s", out_csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
