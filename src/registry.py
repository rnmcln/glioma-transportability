"""Model registry: the set of fitted models evaluated in the analysis.

Keys have the form ``<feature set code>:<model class>``; for example
``B:coxnet`` is the penalised Cox model with the clinical-molecular feature set
and ``R:cox`` is the reference Cox model using the integrated class only.
"""
from __future__ import annotations

import pickle

from .survival_models import MODEL_NAMES

FS_CODE = {"B_clinical_molecular": "B", "A_clinical": "A", "R_class": "R", "R_class_age": "RA"}
CODE_FS = {v: k for k, v in FS_CODE.items()}
MAIN_MODELS = MODEL_NAMES + ["stacked"]
PRIMARY_KEYS = [f"B:{m}" for m in MAIN_MODELS]
CLINICAL_KEYS = [f"A:{m}" for m in MAIN_MODELS]
REFERENCE_KEYS = ["R:cox", "RA:cox"]
ALL_KEYS = PRIMARY_KEYS + CLINICAL_KEYS + REFERENCE_KEYS
BENCHMARK = "B:coxnet"


def load_models(paths, keys=ALL_KEYS) -> dict:
    out = {}
    for k in keys:
        code, name = k.split(":")
        out[k] = pickle.load(open(paths.models / CODE_FS[code] / f"{name}.pkl", "rb"))
    return out


def comparison_pairs():
    """(model, reference) pairs for paired bootstrap differences."""
    pairs = [(k, BENCHMARK) for k in PRIMARY_KEYS if k != BENCHMARK]
    pairs += [(f"B:{m}", f"A:{m}") for m in MAIN_MODELS]
    pairs += [(BENCHMARK, "R:cox"), (BENCHMARK, "RA:cox"), ("B:stacked", "B:cox")]
    return pairs
