"""Harmonise TCGA and CGGA clinical tables to a common analytic schema.

Design principles
-----------------
* Original source labels are never overwritten; every harmonised categorical
  variable retains a paired ``*_original`` column.
* Survival time is expressed in days. TCGA records overall survival in months
  and is converted with the mean Gregorian month length; CGGA records days.
* The integrated "WHO-like" class is reconstructed from IDH, 1p/19q and grade
  where available. It is explicitly *not* a WHO 2021 diagnosis because not all
  defining criteria (for example CDKN2A/B status) are present in both cohorts.
* Inclusion for the primary analysis requires adult age, a positive observed
  survival time, a non-missing event indicator, and primary (non-recurrent)
  tumour status where that status is identifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import DAYS_PER_MONTH, get_logger

log = get_logger("harmonisation")

# Final analytic schema (ordered) ------------------------------------------------
ANALYTIC_COLUMNS = [
    "cohort",
    "source_dataset",
    "patient_id",
    "sample_id",
    "age_years",
    "sex",
    "diagnosis_original",
    "histology_harmonised",
    "grade_original",
    "grade_harmonised",
    "idh_status",
    "idh_mutation_detail",
    "codeletion_1p19q",
    "mgmt_promoter_methylated",
    "primary_recurrent_status",
    "radio_status",
    "chemo_status",
    "os_time_days",
    "os_event",
    "integrated_class_fine",
    "integrated_class_harmonised",
    "included_primary_analysis",
    "exclusion_reason",
]

UNKNOWN = "unknown"


# --- helpers ------------------------------------------------------------------
def _norm(x) -> str:
    if pd.isna(x):
        return UNKNOWN
    return str(x).strip()


def _sex(x) -> str:
    s = _norm(x).lower()
    if s.startswith("f"):
        return "Female"
    if s.startswith("m"):
        return "Male"
    return UNKNOWN


def _idh(x) -> str:
    s = _norm(x).lower()
    if s in {"mutant", "mut", "mutation", "idhmut"}:
        return "mutant"
    if s in {"wildtype", "wt", "wild-type", "idhwt", "wild type"}:
        return "wildtype"
    return UNKNOWN


def _codel(x) -> str:
    s = _norm(x).lower()
    if "non" in s or "intact" in s or s == "noncodel":
        return "noncodeleted"
    if "codel" in s or s == "codel":
        return "codeleted"
    return UNKNOWN


def _mgmt(x) -> str:
    s = _norm(x).lower().replace("-", "").replace("_", "")
    if s.startswith("unmethyl") or s == "unmethylated":
        return "unmethylated"
    if s.startswith("methyl"):
        return "methylated"
    return UNKNOWN


# Grade mappers
def _grade_tcga(x) -> str:
    s = _norm(x).upper().replace("G", "")
    return s if s in {"2", "3", "4"} else UNKNOWN


def _grade_cgga(x) -> str:
    s = _norm(x).upper().replace("WHO", "").strip()
    roman = {"II": "2", "III": "3", "IV": "4", "2": "2", "3": "3", "4": "4"}
    return roman.get(s, UNKNOWN)


# Histology mappers
def _hist_tcga(x) -> str:
    s = _norm(x).lower()
    if "glioblast" in s:
        return "glioblastoma"
    if "oligoastro" in s:
        return "oligoastrocytoma_legacy"
    if "oligodendro" in s:
        return "oligodendroglioma"
    if "astro" in s:
        return "astrocytoma"
    return "diffuse_glioma_NOS"


# CGGA histology codes: optional leading r (recurrent) or s (secondary); A/AA
# astrocytoma, O oligodendroglioma, AO anaplastic oligodendroglioma, OA/AOA
# oligoastrocytoma, GBM glioblastoma.
_CGGA_HIST = {
    "A": "astrocytoma",
    "AA": "astrocytoma",
    "O": "oligodendroglioma",
    "AO": "oligodendroglioma",
    "OA": "oligoastrocytoma_legacy",
    "AOA": "oligoastrocytoma_legacy",
    "GBM": "glioblastoma",
}


def _hist_cgga(x) -> str:
    s = _norm(x)
    if s == UNKNOWN or s == "" or s.lower() == "nan":
        return "diffuse_glioma_NOS"
    core = s
    while core and core[0] in {"r", "s"} and len(core) > 1:
        core = core[1:]
    core = core.upper()
    return _CGGA_HIST.get(core, "diffuse_glioma_NOS")


def _prs_cgga(row) -> str:
    # Prefer the explicit PRS_type field; otherwise infer recurrence from the
    # leading 'r' in the histology code; 's' (secondary) is treated as primary
    # presentation of a secondary tumour.
    prs = _norm(row.get("PRS_type"))
    if prs.lower().startswith("prim"):
        return "Primary"
    if prs.lower().startswith("recur"):
        return "Recurrent"
    if prs.lower().startswith("second"):
        return "Secondary"
    hist = _norm(row.get("Histology"))
    if hist and hist[0] == "r":
        return "Recurrent"
    if hist and hist[0] == "s":
        return "Secondary"
    return UNKNOWN


# --- integrated class ---------------------------------------------------------
def _integrated_fine(idh: str, codel: str, grade: str, hist: str) -> str:
    """Granular WHO-like class including grade where informative."""
    if idh == "wildtype":
        return "glioblastoma_IDHwt"
    if idh == "mutant" and codel == "codeleted":
        g = grade if grade in {"2", "3"} else "NOS"
        return f"oligodendroglioma_IDHmut_codel_G{g}"
    if idh == "mutant" and codel != "codeleted":
        g = grade if grade in {"2", "3", "4"} else "NOS"
        return f"astrocytoma_IDHmut_G{g}"
    return "diffuse_glioma_NOS"


def _integrated_coarse(idh: str, codel: str, grade: str, hist: str) -> str:
    """Coarse WHO-like class used as a model predictor (avoids sparse cells)."""
    if idh == "wildtype":
        return "glioblastoma_IDHwt"
    if idh == "mutant" and codel == "codeleted":
        return "oligodendroglioma_IDHmut_codel"
    if idh == "mutant":
        return "astrocytoma_IDHmut"
    return "diffuse_glioma_NOS"


# --- assembling the analytic frame -------------------------------------------
def _assemble(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    df["integrated_class_fine"] = [
        _integrated_fine(i, c, g, h)
        for i, c, g, h in zip(
            df.idh_status, df.codeletion_1p19q, df.grade_harmonised, df.histology_harmonised
        )
    ]
    df["integrated_class_harmonised"] = [
        _integrated_coarse(i, c, g, h)
        for i, c, g, h in zip(
            df.idh_status, df.codeletion_1p19q, df.grade_harmonised, df.histology_harmonised
        )
    ]
    df = _apply_inclusion(df)
    return df.reindex(columns=ANALYTIC_COLUMNS)


def _apply_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    reasons = []
    included = []
    for _, r in df.iterrows():
        why = []
        age = r.age_years
        if pd.isna(age):
            why.append("age_missing")
        elif age < 18:
            why.append("paediatric")
        if pd.isna(r.os_event):
            why.append("os_event_missing")
        if pd.isna(r.os_time_days):
            why.append("os_time_missing")
        elif r.os_time_days <= 0:
            why.append("os_time_nonpositive")
        if str(r.primary_recurrent_status) in {"Recurrent", "Secondary"}:
            why.append("non_primary_tumour")
        reasons.append(";".join(why) if why else "")
        included.append(len(why) == 0)
    out = df.copy()
    out["exclusion_reason"] = reasons
    out["included_primary_analysis"] = included
    return out


def harmonise_tcga(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["cohort"] = "TCGA"
    df["source_dataset"] = "lgggbm_tcga_pub"
    df["patient_id"] = raw["PATIENT_ID"].astype(str)
    df["sample_id"] = raw.get("SAMPLE_ID", raw["PATIENT_ID"]).astype(str)
    df["age_years"] = pd.to_numeric(raw["AGE"], errors="coerce")
    df["sex"] = raw["SEX"].map(_sex)
    df["diagnosis_original"] = raw["HISTOLOGICAL_DIAGNOSIS"].map(_norm)
    df["histology_harmonised"] = raw["HISTOLOGICAL_DIAGNOSIS"].map(_hist_tcga)
    df["grade_original"] = raw["GRADE"].map(_norm)
    df["grade_harmonised"] = raw["GRADE"].map(_grade_tcga)
    df["idh_status"] = raw["IDH_STATUS"].map(_idh)
    df["idh_mutation_detail"] = raw.get("IDH_CODEL_SUBTYPE", pd.Series([np.nan] * len(raw))).map(_norm)
    df["codeletion_1p19q"] = raw["IDH_1P19Q_SUBTYPE"].map(_codel)
    df["mgmt_promoter_methylated"] = raw["MGMT_PROMOTER_STATUS"].map(_mgmt)
    df["primary_recurrent_status"] = "Primary"  # TCGA pan-glioma are primary resections
    df["radio_status"] = np.nan  # not reliably populated in this TCGA export
    df["chemo_status"] = np.nan
    os_months = pd.to_numeric(raw["OS_MONTHS"], errors="coerce")
    df["os_time_days"] = os_months * DAYS_PER_MONTH
    status = raw["OS_STATUS"].astype(str)
    df["os_event"] = np.where(
        status.str.contains("DECEAS", case=False),
        1.0,
        np.where(status.str.contains("LIVING", case=False), 0.0, np.nan),
    )
    out = _assemble(df)
    log.info("TCGA harmonised: %d rows, %d included in primary analysis",
             len(out), int(out.included_primary_analysis.sum()))
    return out


def harmonise_cgga(raw: pd.DataFrame, release: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df["cohort"] = "CGGA"
    df["source_dataset"] = release
    df["patient_id"] = raw["CGGA_ID"].astype(str)
    df["sample_id"] = raw["CGGA_ID"].astype(str)
    df["age_years"] = pd.to_numeric(raw["Age"], errors="coerce")
    df["sex"] = raw["Gender"].map(_sex)
    df["diagnosis_original"] = raw["Histology"].map(_norm)
    df["histology_harmonised"] = raw["Histology"].map(_hist_cgga)
    df["grade_original"] = raw["Grade"].map(_norm)
    df["grade_harmonised"] = raw["Grade"].map(_grade_cgga)
    df["idh_status"] = raw["IDH_mutation_status"].map(_idh)
    df["idh_mutation_detail"] = raw["IDH_mutation_status"].map(_norm)
    df["codeletion_1p19q"] = raw["1p19q_codeletion_status"].map(_codel)
    df["mgmt_promoter_methylated"] = raw["MGMT_status"].map(_mgmt)
    df["primary_recurrent_status"] = raw.apply(_prs_cgga, axis=1)
    df["radio_status"] = pd.to_numeric(raw.get("Radio_status"), errors="coerce")
    df["chemo_status"] = pd.to_numeric(raw.get("Chemo_status"), errors="coerce")
    df["os_time_days"] = pd.to_numeric(raw["OS"], errors="coerce")
    df["os_event"] = pd.to_numeric(raw["Censor"], errors="coerce")
    out = _assemble(df)
    log.info("CGGA %s harmonised: %d rows, %d included in primary analysis",
             release, len(out), int(out.included_primary_analysis.sum()))
    return out
