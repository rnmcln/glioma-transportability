#!/usr/bin/env python
"""06_train_tcga_models.py - train all model classes in TCGA.

Models (Cox, penalised Cox, random survival forest, gradient-boosted survival
and the stacked ensemble) are trained on the TCGA primary-analysis cohort for
the clinical (A) and clinical-molecular (B) feature sets. Two reference Cox
models (integrated class only; integrated class plus age) are trained to
quantify the discrimination carried by the molecular class. CGGA is not read.
"""
from __future__ import annotations

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
FEATURE_SETS = ["B_clinical_molecular", "A_clinical"]
REFERENCE_SETS = ["R_class", "R_class_age"]


def main() -> int:
    set_global_seed()
    paths = get_paths()
    tcga = pd.read_parquet(paths.data_processed / "analytic_primary_tcga.parquet")
    for fs in FEATURE_SETS:
        outdir = paths.models / fs
        outdir.mkdir(parents=True, exist_ok=True)
        for name in MODEL_NAMES:
            t0 = time.time()
            m = SurvivalModel.fit(name, tcga, fs)
            pickle.dump(m, open(outdir / f"{name}.pkl", "wb"))
            log.info("trained %s (%s) in %.1fs", name, fs, time.time() - t0)
        t0 = time.time()
        st = StackedSurvivalModel.fit(tcga, fs, n_splits=5, n_repeats=5)
        pickle.dump(st, open(outdir / "stacked.pkl", "wb"))
        log.info("trained stacked (%s) in %.1fs", fs, time.time() - t0)
    for fs in REFERENCE_SETS:
        outdir = paths.models / fs
        outdir.mkdir(parents=True, exist_ok=True)
        m = SurvivalModel.fit("cox", tcga, fs)
        pickle.dump(m, open(outdir / "cox.pkl", "wb"))
        log.info("trained reference cox (%s)", fs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
