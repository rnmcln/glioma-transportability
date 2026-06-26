#!/usr/bin/env python
"""run_all.py - reproduce the full analysis end to end.

Long-running, checkpointed steps (training the stacked ensemble, internal CV,
external validation, model updating) are re-invoked until they report
completion, so the pipeline tolerates a capped per-process wall-time budget.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
FS_LIST = ["A_clinical", "B_clinical_molecular"]
CGGA = ["mRNAseq_693", "mRNAseq_325", "mRNA_array_301", "WEseq_286"]


def run(args, until=None, max_tries=12):
    """Run a script; if `until` is set, repeat until the marker appears."""
    for _ in range(max_tries if until else 1):
        out = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
        log = out.stdout + out.stderr
        if until is None or until in log:
            return
    print(f"WARNING: '{' '.join(args)}' did not reach '{until}'")


def main():
    run(["scripts/03_harmonise.py"])
    run(["scripts/04_describe_cohorts.py"])
    run(["scripts/05_cohort_shift.py"])
    for fs in FS_LIST:
        run(["scripts/06_train_tcga_models.py", "--feature-set", fs, "--only", "base"])
        run(["scripts/06_train_tcga_models.py", "--feature-set", fs, "--only", "stacked"])
        run(["scripts/07_internal_validation.py", "--feature-set", fs], until="internal CV complete")
    for fs in FS_LIST:
        for c in CGGA:
            run(["scripts/08_external_validation.py", "--cohort", c, "--feature-set", fs],
                until="external validation complete")
    for c in ["mRNAseq_693", "mRNAseq_325"]:
        run(["scripts/09_model_updating.py", "--cohort", c, "--feature-set", "B_clinical_molecular"],
            until="updating complete")
    run(["scripts/10_sensitivity_analyses.py"])
    run(["scripts/11_figures_tables.py"])
    print("Pipeline complete. See outputs/figures and outputs/tables.")


if __name__ == "__main__":
    main()
