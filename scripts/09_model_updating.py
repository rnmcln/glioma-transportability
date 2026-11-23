#!/usr/bin/env python
"""09_model_updating.py - Stage 0-4 updating ladder in the primary CGGA releases.

Stages 0-3 are evaluated for every clinical-molecular model; Stage 4 (full refit
in CGGA) for the four base learners. A per-fold refit of the complete stacked
ensemble is not performed. Stage 1-4 predictions are held-out (5-fold CV).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.registry import PRIMARY_KEYS, load_models  # noqa: E402
from src.survival_models import MODEL_NAMES, SurvivalModel  # noqa: E402
from src.updating import run_ladder  # noqa: E402
from src.utils import CGGA_PRIMARY_RELEASES, get_logger, get_paths, set_global_seed  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
log = get_logger("09_updating")


def main() -> int:
    set_global_seed()
    paths = get_paths()
    from importlib import import_module
    ext = import_module("08_external_validation")
    models = load_models(paths, PRIMARY_KEYS)
    rows, store = [], {}
    for coh in CGGA_PRIMARY_RELEASES:
        df = ext.load_cohort(paths, coh)
        for key, m in models.items():
            name = key.split(":")[1]
            refit = (lambda d, nm=name: SurvivalModel.fit(nm, d, "B_clinical_molecular")) if name in MODEL_NAMES else None
            lad, preds = run_ladder(df, m, refit_fn=refit, do_stage4=refit is not None)
            lad.insert(0, "model", key); lad.insert(0, "cohort", coh)
            rows.append(lad)
            for s, p in preds.items():
                store[f"{coh}|{key}|{s}"] = p
            log.info("%s %s\n%s", coh, key, lad.drop(columns=["cohort", "model"]).round(3).to_string(index=False))
    pd.concat(rows).to_csv(paths.tables / "updating_ladder.csv", index=False)
    np.savez_compressed(paths.data_processed / "updating_predictions.npz", **store)
    log.info("updating complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
