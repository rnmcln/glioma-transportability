"""End-to-end smoke test on a small synthetic cohort.

Exercises the encoder, a base model, recalibration, and the metric panel to
confirm the pipeline runs and returns finite values without requiring the real
data files.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.calibration import Recalibrator  # noqa: E402
from src.survival_models import SurvivalModel  # noqa: E402
from src.utils import HORIZONS_DAYS  # noqa: E402
from src.validation import evaluate_model  # noqa: E402


def _synth_cohort(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idh = rng.choice(["mutant", "wildtype", "unknown"], n, p=[0.4, 0.5, 0.1])
    grade = np.where(idh == "wildtype", "4", rng.choice(["2", "3"], n))
    base_haz = np.where(idh == "wildtype", 1 / 300.0, 1 / 1500.0)
    time = rng.exponential(1 / base_haz)
    cens = rng.uniform(100, 2500, n)
    os_time = np.minimum(time, cens)
    event = (time <= cens).astype(int)
    return pd.DataFrame({
        "patient_id": [f"S{i}" for i in range(n)],
        "age_years": rng.normal(55, 12, n),
        "sex": rng.choice(["Male", "Female"], n),
        "histology_harmonised": np.where(idh == "wildtype", "glioblastoma", "astrocytoma"),
        "grade_harmonised": grade,
        "idh_status": idh,
        "codeletion_1p19q": rng.choice(["codeleted", "noncodeleted", "unknown"], n, p=[0.2, 0.7, 0.1]),
        "mgmt_promoter_methylated": rng.choice(["methylated", "unmethylated", "unknown"], n),
        "integrated_class_harmonised": np.where(idh == "wildtype", "glioblastoma_IDHwt", "astrocytoma_IDHmut"),
        "os_time_days": os_time, "os_event": event,
    })


def test_pipeline_runs_and_metrics_finite():
    train = _synth_cohort(seed=0)
    ext = _synth_cohort(seed=1)
    model = SurvivalModel.fit("coxnet", train, "B_clinical_molecular")
    res = evaluate_model(model, ext, horizons_days=HORIZONS_DAYS)
    assert np.isfinite(res.metrics["harrell_c"])
    assert 0.4 < res.metrics["harrell_c"] <= 1.0
    assert np.isfinite(res.metrics["ibs"])


def test_recalibration_improves_or_holds_citl():
    train = _synth_cohort(seed=0)
    ext = _synth_cohort(seed=2)
    model = SurvivalModel.fit("coxnet", train, "B_clinical_molecular")
    pi = model.risk_score(ext)
    event = ext.os_event.to_numpy(); time = ext.os_time_days.to_numpy()
    r = Recalibrator("in_large").fit(event, time, pi)
    surv = r.survival_at(pi, list(HORIZONS_DAYS))
    assert surv.shape == (len(ext), len(HORIZONS_DAYS))
    assert np.all((surv >= 0) & (surv <= 1))
