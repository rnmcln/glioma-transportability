#!/usr/bin/env python3
"""Run original analysis stages using already staged TCGA-CDR inputs."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

CONCORDANCE = [
    ("04_describe_cohorts.py",), ("05_cohort_shift.py",), ("06_train_tcga_models.py",),
    ("07_internal_validation.py",), ("08_external_validation.py",),
    ("09_model_updating.py",), ("10_subgroups_decomposition.py",),
    ("11_interpretation_checks.py",), ("12_extensions.py",),
    ("13_expression_signature.py",), ("14_figures_tables.py",),
]
BENCHMARK = [
    ("03_internal_cv.py",), ("03_internal_cv.py", "--unweighted"),
    ("04_external_validation.py",), ("04_external_validation.py", "--unweighted"),
    ("05_survival.py",), ("06_sensitivity.py",), ("07_tables.py",),
    ("10_selection_equivalence.py",), ("11_class_weighting.py",),
    ("12_censoring_weights.py",), ("13_full_cohort.py",), ("14_nested_tuning.py",),
    ("15_seed_variability.py",), ("16_gravendeel_idh.py",),
    ("17_survival_primary.py", "--repeats", "0", "1", "2", "3", "4", "--external"),
    ("17_survival_primary.py", "--criterion", "pll", "--repeats", "0", "1", "2", "3", "4", "--external"),
    ("18_survival_evaluation.py",), ("18_survival_evaluation.py",),
    ("19_within_group.py",), ("20_binary_additions.py",), ("08_figures.py",),
]


def benchmark_cohort_tables(root):
    # Run the existing descriptive stage after prepared input replaces harmonisation.
    import pandas as pd
    import numpy as np

    path = root / "scripts" / "02_build_cohort.py"
    sys.path[:0] = [str(path.parent), str(root)]
    spec = importlib.util.spec_from_file_location("benchmark_cohort_tables", path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    processed = root / "data" / "processed"
    tcga = pd.read_csv(processed / "clinical_tcga.csv")
    if len(tcga) != 663 or int(tcga.os_event.sum()) != 245 or not np.isfinite(tcga.os_months).all():
        raise ValueError("Benchmark TCGA-CDR stage must be prepared first")
    if not (processed / "harmonisation_summary.csv").exists():
        raise FileNotFoundError("Staged harmonisation summary is missing")
    summary = pd.read_csv(processed / "harmonisation_summary.csv")
    stage.harmonise_all = lambda *, save: (None, summary)
    stage.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=["concordance", "benchmark"])
    parser.add_argument("repository", type=Path)
    parser.add_argument("--cohort-tables-only", action="store_true")
    args = parser.parse_args()
    root = args.repository.resolve()
    if args.pipeline == "benchmark":
        benchmark_cohort_tables(root)
        if args.cohort_tables_only:
            return
        stages = BENCHMARK
    else:
        stages = CONCORDANCE[:1] if args.cohort_tables_only else CONCORDANCE

    seed = "20260628" if args.pipeline == "benchmark" else "20240626"
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONHASHSEED=seed)
    evaluation_seen = False
    for script, *rest in stages:
        print(f"== {script} {' '.join(rest)}", flush=True)
        current_env = env.copy()
        current_env.pop("SURV_CRITERION", None)
        if script == "18_survival_evaluation.py":
            if evaluation_seen:
                current_env["SURV_CRITERION"] = "pll"
            evaluation_seen = True
        subprocess.run([sys.executable, str(root / "scripts" / script), *rest],
                       cwd=root, env=current_env, check=True)


if __name__ == "__main__":
    main()
