"""Tests for harmonisation logic and the analytic schema."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.harmonisation import (ANALYTIC_COLUMNS, _hist_cgga, _integrated_coarse,  # noqa: E402
                               _prs_cgga, harmonise_cgga, harmonise_tcga)
from src.utils import DAYS_PER_MONTH


def _tcga_raw():
    return pd.DataFrame({
        "PATIENT_ID": ["P1", "P2", "P3"],
        "SAMPLE_ID": ["P1-01", "P2-01", "P3-01"],
        "AGE": [55.0, 16.0, 60.0],
        "SEX": ["Male", "Female", "Male"],
        "HISTOLOGICAL_DIAGNOSIS": ["glioblastoma", "astrocytoma", "oligodendroglioma"],
        "GRADE": ["G4", "G2", "G3"],
        "IDH_STATUS": ["WT", "Mutant", "Mutant"],
        "IDH_CODEL_SUBTYPE": ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"],
        "IDH_1P19Q_SUBTYPE": ["Non-codel", "Non-codel", "Codel"],
        "MGMT_PROMOTER_STATUS": ["Unmethylated", "Methylated", "Methylated"],
        "OS_MONTHS": [12.0, 100.0, 50.0],
        "OS_STATUS": ["1:DECEASED", "0:LIVING", "1:DECEASED"],
    })


def test_schema_and_os_conversion():
    h = harmonise_tcga(_tcga_raw())
    assert list(h.columns) == ANALYTIC_COLUMNS
    # months -> days
    assert np.isclose(h.loc[0, "os_time_days"], 12.0 * DAYS_PER_MONTH)
    assert h.loc[0, "os_event"] == 1.0 and h.loc[1, "os_event"] == 0.0


def test_inclusion_excludes_paediatric():
    h = harmonise_tcga(_tcga_raw())
    # P2 is 16 years -> excluded as paediatric
    assert not bool(h.loc[1, "included_primary_analysis"])
    assert "paediatric" in h.loc[1, "exclusion_reason"]
    assert bool(h.loc[0, "included_primary_analysis"])


def test_integrated_class():
    assert _integrated_coarse("wildtype", "noncodeleted", "4", "glioblastoma") == "glioblastoma_IDHwt"
    assert _integrated_coarse("mutant", "codeleted", "2", "oligodendroglioma") == "oligodendroglioma_IDHmut_codel"
    assert _integrated_coarse("mutant", "noncodeleted", "3", "astrocytoma") == "astrocytoma_IDHmut"
    assert _integrated_coarse("unknown", "unknown", "unknown", "x") == "diffuse_glioma_NOS"


def test_cgga_histology_and_recurrence():
    # leading r denotes recurrence; secondary GBM treated as primary presentation
    assert _hist_cgga("rGBM") == "glioblastoma"
    assert _hist_cgga("AA") == "astrocytoma"
    assert _hist_cgga("AO") == "oligodendroglioma"
    assert _prs_cgga({"PRS_type": "Recurrent", "Histology": "rGBM"}) == "Recurrent"
    assert _prs_cgga({"PRS_type": np.nan, "Histology": "rAA"}) == "Recurrent"
    assert _prs_cgga({"PRS_type": "Primary", "Histology": "GBM"}) == "Primary"


def test_cgga_recurrent_excluded_from_primary():
    raw = pd.DataFrame({
        "CGGA_ID": ["C1", "C2"], "PRS_type": ["Primary", "Recurrent"],
        "Histology": ["GBM", "rGBM"], "Grade": ["WHO IV", "WHO IV"],
        "Gender": ["Male", "Female"], "Age": [50, 45],
        "OS": [400, 600], "Censor": [1, 0],
        "IDH_mutation_status": ["Wildtype", "Wildtype"],
        "1p19q_codeletion_status": ["Non-codel", "Non-codel"],
        "MGMT_status": ["methylated", "un-methylated"],
        "Radio_status": [1, 1], "Chemo_status": [1, 0],
    })
    h = harmonise_cgga(raw, "test_release")
    assert bool(h.loc[0, "included_primary_analysis"])
    assert not bool(h.loc[1, "included_primary_analysis"])
    assert "non_primary_tumour" in h.loc[1, "exclusion_reason"]
