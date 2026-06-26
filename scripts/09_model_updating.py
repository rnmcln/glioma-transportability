#!/usr/bin/env python
"""09_model_updating.py - evaluate the Stage 0-4 updating ladder on a CGGA release.

Stages 0-3 are run for every model; Stage 4 (full refit) is run for the base
learners (a stacked refit per fold is omitted for cost and noted as such).
Checkpointed per model. Run until 'updating complete'.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.survival_models import MODEL_NAMES, SurvivalModel  # noqa: E402
from src.updating import run_ladder  # noqa: E402
from src.utils import get_logger, get_paths, set_global_seed  # noqa: E402

log = get_logger("09_updating")
ALL_MODELS = MODEL_NAMES + ["stacked"]


def _load_cohort(paths, release):
    key = release.lower().replace("_", "")
    df = pd.read_parquet(paths.data_processed / f"cgga_{key}_harmonised.parquet")
    return df[df.included_primary_analysis].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--feature-set", default="B_clinical_molecular")
    ap.add_argument("--budget", type=float, default=36.0)
    args = ap.parse_args()
    set_global_seed()
    paths = get_paths()
    df = _load_cohort(paths, args.cohort)
    fs = args.feature_set
    mdir = paths.models / fs

    partial = paths.data_processed / f"upd_{args.cohort}_{fs}_partial.pkl"
    done = pickle.load(open(partial, "rb")) if partial.exists() else {}

    t0 = time.time()
    for name in ALL_MODELS:
        if name in done:
            continue
        if time.time() - t0 > args.budget:
            break
        frozen = pickle.load(open(mdir / f"{name}.pkl", "rb"))
        if name in MODEL_NAMES:
            refit_fn = (lambda d, nm=name: SurvivalModel.fit(nm, d, fs))
            do4 = True
        else:
            refit_fn, do4 = None, False  # stacked Stage 4 omitted (cost)
        ladder = run_ladder(df, frozen, refit_fn=refit_fn, feature_set=fs, do_stage4=do4)
        ladder.insert(0, "model", name)
        done[name] = ladder
        pickle.dump(done, open(partial, "wb"))
        log.info("%s ladder done (%.1fs)\n%s", name, time.time() - t0,
                 ladder[["stage", "harrell_c", "citl_primary", "cal_slope", "ibs"]].round(3).to_string(index=False))

    if len(done) < len(ALL_MODELS):
        log.info("checkpoint: %d/%d models; re-run", len(done), len(ALL_MODELS))
        return 0

    out = pd.concat([done[m] for m in ALL_MODELS], ignore_index=True)
    out.insert(0, "cohort", args.cohort)
    out.to_csv(paths.tables / f"table6_updating_{args.cohort}_{fs}.csv", index=False)
    log.info("updating complete -> table6_updating_%s_%s.csv", args.cohort, fs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
