"""Loaders for the raw TCGA (cBioPortal datahub) and CGGA clinical files.

Both loaders return the raw tables with their original column names preserved.
Harmonisation to a common schema is performed separately in ``harmonisation.py``
so that the mapping from source columns to analytic variables is auditable.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from .utils import Paths, get_logger

log = get_logger("data_io")

# cBioPortal clinical files carry four leading metadata rows prefixed with '#';
# the fifth row is the machine-readable header.
_CBIO_SKIPROWS = 4


def load_tcga_raw(paths: Paths) -> pd.DataFrame:
    """Merge the TCGA patient- and sample-level clinical tables on PATIENT_ID.

    The merged cohort has exactly one sample per patient, so the join is 1:1.
    """
    pat = pd.read_csv(
        paths.tcga_raw / "data_clinical_patient.txt", sep="\t", skiprows=_CBIO_SKIPROWS
    )
    sam = pd.read_csv(
        paths.tcga_raw / "data_clinical_sample.txt", sep="\t", skiprows=_CBIO_SKIPROWS
    )
    n_per_patient = sam.groupby("PATIENT_ID").size().max()
    if n_per_patient != 1:
        log.warning("TCGA has up to %d samples per patient; index rule will apply.", n_per_patient)
    merged = pat.merge(sam, on="PATIENT_ID", how="inner", suffixes=("", "_sample"))
    log.info("TCGA raw merged: %d rows", len(merged))
    return merged


def _find_cgga_clinical(paths: Paths, release: str) -> Path:
    pattern = str(paths.cgga_raw / release / "*clinical*.txt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CGGA clinical file for release {release} ({pattern})")
    return Path(matches[0])


def load_cgga_raw(paths: Paths, release: str) -> pd.DataFrame:
    """Load a single CGGA release clinical table with original column names.

    Column names vary slightly across releases (for example ``IDH_mut_status``
    in WEseq_286 versus ``IDH_mutation_status`` elsewhere). These are normalised
    to canonical names here so downstream harmonisation sees a stable schema.
    """
    f = _find_cgga_clinical(paths, release)
    df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "IDH_mut_status": "IDH_mutation_status",
        "1p19q_Codeletion_status": "1p19q_codeletion_status",
        "Censor (alive=0; dead=1)": "Censor",
        "Radio_status (treated=1;un-treated=0)": "Radio_status",
        "Chemo_status (TMZ treated=1;un-treated=0)": "Chemo_status",
        "MGMTp_methylation_status": "MGMT_status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["source_dataset"] = release
    log.info("CGGA %s raw: %d rows", release, len(df))
    return df
