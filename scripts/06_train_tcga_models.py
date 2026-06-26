#!/usr/bin/env python
"""06_train_tcga_models.py - train all model classes in TCGA for one feature set.

Usage:
    python scripts/06_train_tcga_models.py --feature-set B_clinical_molecular

Models (Cox, penalised Cox, RSF, gradient-boosted survival, stacked ensemble)
are trained exclusively on the TCGA primary-analysis cohort and pickled to
outputs/models/<feature_set>/. CGGA is never touched here.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.stacking import StackedSurvivalModel  # noqa: E402
from src.survival_models import MODEL_NAMES, SurvivalModel  # noqa: E402
from src.utils import get_logger, get_paths, set_global_seed  # noqa: E402

log = get_logger("06_train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="B_clinical_molecular")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--only", choices=["all", "base", "stacked"], default="all",
                    help="Train only base learners, only the stacked ensemble, or all.")
    args = ap.parse_args()

    set_global_seed()
    paths = get_paths()
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet")
    fs = args.feature_set
    outdir = paths.models / fs
    outdir.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "base"):
        for name in MODEL_NAMES:
            t0 = time.time()
            m = SurvivalModel.fit(name, tcga, fs)
            with open(outdir / f"{name}.pkl", "wb") as f:
                pickle.dump(m, f)
            log.info("trained %s (%s) in %.1fs", name, fs, time.time() - t0)

    if args.only in ("all", "stacked"):
        t0 = time.time()
        st = StackedSurvivalModel.fit(tcga, fs, n_splits=args.n_splits, n_repeats=args.n_repeats)
        with open(outdir / "stacked.pkl", "wb") as f:
            pickle.dump(st, f)
        log.info("trained stacked (%s) in %.1fs", fs, time.time() - t0)
    log.info("saved models -> %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
