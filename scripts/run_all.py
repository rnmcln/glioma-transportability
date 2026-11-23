#!/usr/bin/env python
"""run_all.py - reproduce the full analysis end to end (stages 03-12).

Stages 01 and 02 download the public source files; run them first if
data_raw/ is empty. Each stage is a separate process so that a failure stops
the pipeline at the stage concerned.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGES = [
    "03_harmonise.py",
    "04_describe_cohorts.py",
    "05_cohort_shift.py",
    "06_train_tcga_models.py",
    "07_internal_validation.py",
    "08_external_validation.py",
    "09_model_updating.py",
    "10_subgroups_decomposition.py",
    "11_interpretation_checks.py",
    "12_figures_tables.py",
]


def main() -> int:
    for s in STAGES:
        print(f"== {s}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT)
        if r.returncode != 0:
            print(f"stage {s} failed with exit code {r.returncode}")
            return r.returncode
    print("pipeline complete; outputs in outputs/tables and outputs/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
